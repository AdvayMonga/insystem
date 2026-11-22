use std::time::{Duration, Instant};

use crate::kv::BlockManager;
use crate::plan::Executor;
use crate::request::{Request, RequestId, RequestOutput};
use crate::scheduler::{Scheduler, SchedulerConfig};

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_model_len: usize,
    pub enable_prefix_caching: bool,
    pub eos_token_ids: Vec<u32>,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self { max_num_seqs: 64, max_num_batched_tokens: 512, max_model_len: 4096, enable_prefix_caching: true, eos_token_ids: vec![] }
    }
}

#[derive(Clone, Debug, Default)]
pub struct EngineStats {
    pub steps: u64,
    pub scheduled_tokens: u64,
    pub prefill_tokens: u64,
    pub sampled_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub prompt_tokens: u64,
    pub exec_time: Duration,
    pub sched_time: Duration,
}

pub struct Engine<E: Executor> {
    pub sched: Scheduler,
    pub exec: E,
    pub stats: EngineStats,
}

impl<E: Executor> Engine<E> {
    pub fn new(cfg: EngineConfig, exec: E) -> anyhow::Result<Self> {
        let spec = exec.cache_spec();
        let capacity = spec.num_blocks * spec.block_size;
        anyhow::ensure!(capacity >= cfg.max_model_len, "KV cache holds {capacity} tokens, below max_model_len {}", cfg.max_model_len);
        let kv = BlockManager::new(spec.num_blocks, spec.block_size, cfg.enable_prefix_caching);
        let sched = Scheduler::new(
            SchedulerConfig {
                max_num_seqs: cfg.max_num_seqs,
                max_num_batched_tokens: cfg.max_num_batched_tokens,
                max_model_len: cfg.max_model_len,
                eos_token_ids: cfg.eos_token_ids.clone(),
            },
            kv,
        );
        Ok(Self { sched, exec, stats: EngineStats::default() })
    }

    pub fn add_request(&mut self, req: Request) -> anyhow::Result<()> {
        self.stats.prompt_tokens += req.prompt.len() as u64;
        self.sched.add(req)
    }

    pub fn abort(&mut self, id: RequestId) {
        self.sched.abort(id);
    }

    pub fn has_work(&self) -> bool {
        self.sched.has_work() || !self.sched.aborted.is_empty()
    }

    /// Runs one scheduling step and one forward pass.
    pub fn step(&mut self) -> anyhow::Result<Vec<RequestOutput>> {
        let t0 = Instant::now();
        let plan = self.sched.schedule();
        let t1 = Instant::now();
        if plan.seqs.is_empty() {
            return Ok(std::mem::take(&mut self.sched.aborted));
        }
        let out = self.exec.execute(&plan)?;
        let t2 = Instant::now();
        anyhow::ensure!(out.sampled.len() == plan.num_sampled(), "executor sampled {} rows, plan asked {}", out.sampled.len(), plan.num_sampled());
        let results = self.sched.update(&out);
        let t3 = Instant::now();
        self.stats.steps += 1;
        self.stats.scheduled_tokens += plan.num_tokens() as u64;
        self.stats.prefill_tokens += plan.num_prefill_tokens as u64;
        self.stats.sampled_tokens += out.sampled.len() as u64;
        self.stats.cached_prompt_tokens += results.iter().map(|r| r.num_cached_tokens as u64).sum::<u64>();
        self.stats.exec_time += t2 - t1;
        self.stats.sched_time += (t1 - t0) + (t3 - t2);
        Ok(results)
    }
}
