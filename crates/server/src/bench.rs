//! Offline throughput: every request is submitted at once and the engine runs until all finish,
//! as `vllm bench throughput` does. Prompts are random token ids; lengths are drawn uniformly
//! from `[range_ratio * len, len]`, as InferenceX draws them.

use std::io::Write;
use std::time::Instant;

use clap::Args;
use insystem_core::{Request, SamplingParams};
use serde_json::json;

use crate::{build_engine, EngineArgs};

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    #[arg(long, default_value_t = 32)]
    pub num_prompts: usize,
    #[arg(long, default_value_t = 256)]
    pub input_len: usize,
    #[arg(long, default_value_t = 128)]
    pub output_len: usize,
    #[arg(long, default_value_t = 1.0)]
    pub range_ratio: f64,
    /// Tokens of one prefix shared by every prompt (counted inside input_len).
    #[arg(long, default_value_t = 0)]
    pub prefix_len: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Append the result as one JSON line to this file.
    #[arg(long)]
    pub out: Option<std::path::PathBuf>,
}

pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + (self.next() as usize) % (hi - lo + 1)
    }
}

pub fn run(a: BenchArgs) -> anyhow::Result<()> {
    let mut eng = build_engine(&a.engine)?;
    let vocab = eng.exec.spec.vocab_size.min(150_000) as u64;
    let mut rng = Lcg(a.seed + 1);
    let lo = |n: usize| ((n as f64 * a.range_ratio).round() as usize).max(1).min(n);
    let prefix: Vec<u32> = (0..a.prefix_len).map(|_| (rng.next() % vocab) as u32).collect();
    let mut reqs = Vec::new();
    for i in 0..a.num_prompts {
        let il = rng.range(lo(a.input_len), a.input_len).max(a.prefix_len + 1);
        let ol = rng.range(lo(a.output_len), a.output_len);
        let mut p = prefix.clone();
        p.extend((a.prefix_len..il).map(|_| (rng.next() % vocab) as u32));
        reqs.push(Request { id: i as u64 + 1, prompt: p, params: SamplingParams { max_tokens: ol as u32, ignore_eos: true, temperature: 0.0, ..Default::default() } });
    }
    let n_in: usize = reqs.iter().map(|r| r.prompt.len()).sum();

    // Warm up: one short request so first-touch page faults are not measured.
    eng.add_request(Request { id: u64::MAX, prompt: vec![1, 2, 3, 4], params: SamplingParams { max_tokens: 2, ignore_eos: true, ..Default::default() } })?;
    while eng.has_work() {
        eng.step()?;
    }
    eng.stats = Default::default();
    if let Some(p) = eng.exec.profile.as_mut() {
        *p = Default::default();
    }
    eng.sched.kv.reset_prefix_cache();

    let t0 = Instant::now();
    for r in reqs {
        eng.add_request(r)?;
    }
    let mut n_out = 0usize;
    let mut ttft_sum = 0.0;
    let mut seen = std::collections::HashSet::new();
    while eng.has_work() {
        for o in eng.step()? {
            n_out += o.new_tokens.len();
            if seen.insert(o.id) {
                ttft_sum += t0.elapsed().as_secs_f64();
            }
        }
    }
    let el = t0.elapsed().as_secs_f64();
    let st = &eng.stats;
    let res = json!({
        "model": eng.exec.spec.name,
        "num_prompts": a.num_prompts,
        "input_len": a.input_len, "output_len": a.output_len, "range_ratio": a.range_ratio, "prefix_len": a.prefix_len,
        "max_num_seqs": a.engine.max_num_seqs, "max_num_batched_tokens": a.engine.max_num_batched_tokens,
        "threads": rayon::current_num_threads(),
        "kernels": {"matmul": eng.exec.kernels.matmul.name, "attention": eng.exec.kernels.attention.name},
        "elapsed_s": el,
        "input_tokens": n_in, "output_tokens": n_out,
        "total_tok_s": (n_in + n_out) as f64 / el,
        "output_tok_s": n_out as f64 / el,
        "requests_s": a.num_prompts as f64 / el,
        "mean_ttft_s": ttft_sum / a.num_prompts as f64,
        "steps": st.steps,
        "mean_tokens_per_step": st.scheduled_tokens as f64 / st.steps.max(1) as f64,
        "cached_prompt_tokens": st.cached_prompt_tokens,
        "computed_tokens": st.scheduled_tokens,
        "exec_s": st.exec_time.as_secs_f64(), "sched_s": st.sched_time.as_secs_f64(),
        "preemptions": eng.sched.num_preemptions,
        "profile": eng.exec.profile.as_ref().map(|p| p.to_json()),
    });
    eprintln!(
        "{} prompts: {:.1}s, total {:.1} tok/s, output {:.1} tok/s, {:.1} tokens/step, {} preemptions",
        a.num_prompts, el, res["total_tok_s"].as_f64().unwrap(), res["output_tok_s"].as_f64().unwrap(), res["mean_tokens_per_step"].as_f64().unwrap(), eng.sched.num_preemptions
    );
    if let Some(p) = &eng.exec.profile {
        eprint!("{}", p.report());
    }
    println!("{res}");
    if let Some(path) = &a.out {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(f, "{res}")?;
    }
    Ok(())
}
