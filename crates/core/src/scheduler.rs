//! Continuous-batching scheduler with one token budget for prefill and decode.
//!
//! Each step every sequence wants `len(tokens) - num_computed` more tokens computed: one for a
//! decoding sequence, the rest of the prompt for a prefilling one. Running sequences are served
//! first, in admission order, then waiting sequences are admitted while the token budget and the
//! sequence cap allow. A prompt longer than the remaining budget is computed in chunks over
//! several steps. When the cache cannot grow a running sequence, the most recently admitted
//! sequence is preempted: its blocks are released and it is recomputed from its tokens later.

use std::collections::VecDeque;

use crate::kv::{chain_hash, BlockManager};
use crate::plan::{BatchPlan, SamplerSpec, SeqSlice, StepOutput};
use crate::request::{FinishReason, Request, RequestId, RequestOutput, SamplingParams};

#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_model_len: usize,
    pub eos_token_ids: Vec<u32>,
}

struct Seq {
    id: RequestId,
    tokens: Vec<u32>,
    num_prompt: usize,
    num_computed: usize,
    blocks: Vec<u32>,
    hashes: Vec<u64>,
    params: SamplingParams,
    num_cached_tokens: usize,
    reported_cached: bool,
}

impl Seq {
    fn num_generated(&self) -> usize {
        self.tokens.len() - self.num_prompt
    }
}

pub struct Scheduler {
    pub cfg: SchedulerConfig,
    pub kv: BlockManager,
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
    scheduled: Vec<(usize, usize)>,
    step: u64,
    pub num_preemptions: u64,
    pub aborted: Vec<RequestOutput>,
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

impl Scheduler {
    pub fn new(cfg: SchedulerConfig, kv: BlockManager) -> Self {
        Self { cfg, kv, waiting: VecDeque::new(), running: Vec::new(), scheduled: Vec::new(), step: 0, num_preemptions: 0, aborted: Vec::new() }
    }

    pub fn add(&mut self, req: Request) -> anyhow::Result<()> {
        anyhow::ensure!(!req.prompt.is_empty(), "empty prompt");
        anyhow::ensure!(
            req.prompt.len() < self.cfg.max_model_len,
            "prompt of {} tokens does not fit max_model_len {}",
            req.prompt.len(),
            self.cfg.max_model_len
        );
        let blocks_needed = (req.prompt.len() + 1).div_ceil(self.kv.block_size);
        anyhow::ensure!(blocks_needed <= self.kv.num_blocks(), "prompt does not fit the KV cache");
        self.waiting.push_back(Seq {
            id: req.id,
            num_prompt: req.prompt.len(),
            tokens: req.prompt,
            num_computed: 0,
            blocks: Vec::new(),
            hashes: Vec::new(),
            params: req.params,
            num_cached_tokens: 0,
            reported_cached: false,
        });
        Ok(())
    }

    pub fn abort(&mut self, id: RequestId) {
        if let Some(i) = self.waiting.iter().position(|s| s.id == id) {
            let s = self.waiting.remove(i).unwrap();
            self.kv.release(&s.blocks);
            self.aborted.push(Self::finish_output(&s, FinishReason::Abort));
        } else if let Some(i) = self.running.iter().position(|s| s.id == id) {
            let s = self.running.remove(i);
            self.kv.release(&s.blocks);
            self.aborted.push(Self::finish_output(&s, FinishReason::Abort));
        }
    }

    fn finish_output(s: &Seq, r: FinishReason) -> RequestOutput {
        RequestOutput { id: s.id, new_tokens: vec![], logprobs: vec![], finished: Some(r), num_prompt_tokens: s.num_prompt, num_cached_tokens: s.num_cached_tokens }
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }
    pub fn num_running(&self) -> usize {
        self.running.len()
    }
    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    fn preempt_last(&mut self) {
        let mut s = self.running.pop().expect("preempt with nothing running");
        self.kv.release(&s.blocks);
        s.blocks.clear();
        s.hashes.clear();
        s.num_computed = 0;
        self.num_preemptions += 1;
        self.waiting.push_front(s);
    }

    fn blocks_for(&self, n_tokens: usize) -> usize {
        n_tokens.div_ceil(self.kv.block_size)
    }

    /// Picks this step's work and returns the plan. An empty plan means nothing to do.
    pub fn schedule(&mut self) -> BatchPlan {
        self.scheduled.clear();
        let mut budget = self.cfg.max_num_batched_tokens;
        let mut preempted = false;

        let mut i = 0;
        while i < self.running.len() && budget > 0 {
            let s = &self.running[i];
            let q = (s.tokens.len() - s.num_computed).min(budget);
            let need = self.blocks_for(s.num_computed + q).saturating_sub(s.blocks.len());
            let mut fits = true;
            while !self.kv.can_allocate(need) {
                preempted = true;
                if self.running.len() - 1 == i {
                    self.preempt_last();
                    fits = false;
                    break;
                }
                self.preempt_last();
            }
            if !fits {
                break;
            }
            let new = self.kv.allocate(need).unwrap();
            self.running[i].blocks.extend(new);
            self.scheduled.push((i, q));
            budget -= q;
            i += 1;
        }

        while !preempted && budget > 0 && self.running.len() < self.cfg.max_num_seqs && !self.waiting.is_empty() {
            let s = self.waiting.front_mut().unwrap();
            let mut acquired = 0;
            if s.num_computed == 0 && s.blocks.is_empty() {
                let (blocks, hashes) = self.kv.acquire_prefix(&s.tokens);
                acquired = blocks.len();
                s.num_computed = blocks.len() * self.kv.block_size;
                if !s.reported_cached {
                    s.num_cached_tokens = s.num_computed;
                }
                s.blocks = blocks;
                s.hashes = hashes;
            }
            let q = (s.tokens.len() - s.num_computed).min(budget);
            let need = (s.num_computed + q).div_ceil(self.kv.block_size).saturating_sub(s.blocks.len());
            if !self.kv.can_allocate(need) {
                if acquired > 0 {
                    let blocks = std::mem::take(&mut s.blocks);
                    s.hashes.clear();
                    s.num_computed = 0;
                    self.kv.release(&blocks);
                }
                break;
            }
            let new = self.kv.allocate(need).unwrap();
            let mut s = self.waiting.pop_front().unwrap();
            s.blocks.extend(new);
            self.running.push(s);
            self.scheduled.push((self.running.len() - 1, q));
            budget -= q;
        }

        self.build_plan()
    }

    fn build_plan(&mut self) -> BatchPlan {
        let bs = self.kv.block_size;
        let mut plan = BatchPlan { step: self.step, ..Default::default() };
        self.step += 1;
        let total: usize = self.scheduled.iter().map(|&(_, q)| q).sum();
        plan.token_ids.reserve(total);
        plan.positions.reserve(total);
        plan.slot_mapping.reserve(total);
        for &(idx, q) in &self.scheduled {
            let s = &self.running[idx];
            let start = s.num_computed;
            let q_start = plan.token_ids.len() as u32;
            for p in start..start + q {
                plan.token_ids.push(s.tokens[p]);
                plan.positions.push(p as u32);
                plan.slot_mapping.push(s.blocks[p / bs] * bs as u32 + (p % bs) as u32);
            }
            let bt_start = plan.block_tables.len() as u32;
            let n_blocks = (start + q).div_ceil(bs);
            plan.block_tables.extend_from_slice(&s.blocks[..n_blocks]);
            let sample = if start + q == s.tokens.len() {
                let seed = s.params.seed.unwrap_or(s.id);
                Some(SamplerSpec {
                    temperature: s.params.temperature,
                    top_k: s.params.top_k,
                    top_p: s.params.top_p,
                    rng_seed: mix64(seed ^ mix64(s.num_generated() as u64 + 1)),
                })
            } else {
                None
            };
            if q > 1 {
                plan.num_prefill_tokens += q as u32;
            }
            plan.seqs.push(SeqSlice {
                seq_id: s.id,
                q_start,
                q_len: q as u32,
                ctx_len: (start + q) as u32,
                bt_start,
                bt_len: n_blocks as u32,
                sample,
            });
        }
        plan
    }

    /// Applies the executor's output to the sequences scheduled by the last [`Self::schedule`].
    pub fn update(&mut self, out: &StepOutput) -> Vec<RequestOutput> {
        let bs = self.kv.block_size;
        let mut results = std::mem::take(&mut self.aborted);
        let mut sampled = out.sampled.iter();
        let mut finished_idx = Vec::new();
        let scheduled = std::mem::take(&mut self.scheduled);
        for &(idx, q) in &scheduled {
            let s = &mut self.running[idx];
            s.num_computed += q;
            let full = s.num_computed / bs;
            while s.hashes.len() < full {
                let k = s.hashes.len();
                let parent = if k == 0 { 0 } else { s.hashes[k - 1] };
                let h = chain_hash(parent, &s.tokens[k * bs..(k + 1) * bs]);
                s.hashes.push(h);
                self.kv.register(s.blocks[k], h);
            }
            if s.num_computed < s.tokens.len() {
                continue;
            }
            let smp = sampled.next().expect("executor returned too few samples");
            s.tokens.push(smp.token);
            let mut finished = None;
            let is_eos = self.cfg.eos_token_ids.contains(&smp.token);
            if (is_eos && !s.params.ignore_eos) || s.params.stop_token_ids.contains(&smp.token) {
                finished = Some(FinishReason::Stop);
            } else if s.num_generated() >= s.params.max_tokens as usize || s.tokens.len() >= self.cfg.max_model_len {
                finished = Some(FinishReason::Length);
            }
            let cached = if s.reported_cached { 0 } else { s.num_cached_tokens };
            s.reported_cached = true;
            results.push(RequestOutput {
                id: s.id,
                new_tokens: vec![smp.token],
                logprobs: vec![smp.logprob],
                finished,
                num_prompt_tokens: s.num_prompt,
                num_cached_tokens: cached,
            });
            if finished.is_some() {
                finished_idx.push(idx);
            }
        }
        self.scheduled = scheduled;
        self.scheduled.clear();
        if !finished_idx.is_empty() {
            finished_idx.sort_unstable();
            for &idx in finished_idx.iter().rev() {
                let s = self.running.remove(idx);
                self.kv.release(&s.blocks);
            }
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(blocks: usize, budget: usize) -> Scheduler {
        Scheduler::new(
            SchedulerConfig { max_num_seqs: 8, max_num_batched_tokens: budget, max_model_len: 1024, eos_token_ids: vec![0] },
            BlockManager::new(blocks, 4, true),
        )
    }

    fn req(id: u64, n: usize, max_tokens: u32) -> Request {
        Request { id, prompt: (1..=n as u32).collect(), params: SamplingParams { max_tokens, ignore_eos: true, ..Default::default() } }
    }

    fn run_to_end(s: &mut Scheduler) -> Vec<RequestOutput> {
        let mut all = vec![];
        for _ in 0..1000 {
            let plan = s.schedule();
            if plan.seqs.is_empty() {
                break;
            }
            let out = StepOutput { sampled: vec![crate::Sampled { token: 7, logprob: 0.0 }; plan.num_sampled()] };
            all.extend(s.update(&out));
        }
        all
    }

    #[test]
    fn chunked_prefill_then_decode() {
        let mut s = sched(64, 8);
        s.add(req(1, 20, 3)).unwrap();
        let p = s.schedule();
        assert_eq!(p.num_tokens(), 8);
        assert!(p.seqs[0].sample.is_none());
        let outs = run_to_end(&mut s);
        let toks: usize = outs.iter().map(|o| o.new_tokens.len()).sum();
        assert_eq!(toks, 3);
        assert_eq!(outs.last().unwrap().finished, Some(FinishReason::Length));
        assert_eq!(s.kv.num_free(), 64);
    }

    #[test]
    fn prefix_cache_hit_on_second_request() {
        let mut s = sched(64, 64);
        s.add(req(1, 17, 1)).unwrap();
        run_to_end(&mut s);
        s.add(req(2, 17, 1)).unwrap();
        let outs = run_to_end(&mut s);
        assert_eq!(outs[0].num_cached_tokens, 16);
    }

    #[test]
    fn preemption_recovers() {
        // 6 blocks of 4 tokens; two requests that each grow to 12 tokens need 3 blocks + growth.
        let mut s = sched(6, 64);
        s.add(req(1, 9, 10)).unwrap();
        s.add(req(2, 9, 10)).unwrap();
        let outs = run_to_end(&mut s);
        for id in [1, 2] {
            let n: usize = outs.iter().filter(|o| o.id == id).map(|o| o.new_tokens.len()).sum();
            assert_eq!(n, 10);
        }
        assert!(s.num_preemptions > 0);
        assert_eq!(s.kv.num_free(), 6);
    }
}
