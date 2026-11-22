//! Equivalence against a reference (Hugging Face transformers, fp32) written by
//! `tools/reference.py`: greedy tokens must match, except after a step where the reference's top
//! two log-probabilities are within `tie_margin` (a near-tie either side may take); log-probs of
//! matched tokens must be within `logprob_tol`. All prompts run in one batch.

use std::path::PathBuf;

use clap::Args;
use insystem_core::{Request, SamplingParams};
use serde::Deserialize;
use serde_json::json;

use crate::{build_engine, EngineArgs};

#[derive(Args, Debug)]
pub struct CheckArgs {
    #[command(flatten)]
    pub engine: EngineArgs,
    #[arg(long)]
    pub reference: PathBuf,
    #[arg(long, default_value_t = 0.05)]
    pub logprob_tol: f32,
    #[arg(long, default_value_t = 0.02)]
    pub tie_margin: f32,
}

#[derive(Deserialize)]
struct RefOut {
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
    margins: Vec<f32>,
}

#[derive(Deserialize)]
struct Reference {
    prompts: Vec<Vec<u32>>,
    outputs: Vec<RefOut>,
}

pub fn run(a: CheckArgs) -> anyhow::Result<()> {
    let r: Reference = serde_json::from_str(&std::fs::read_to_string(&a.reference)?)?;
    let mut eng = build_engine(&a.engine)?;
    for (i, (p, o)) in r.prompts.iter().zip(&r.outputs).enumerate() {
        eng.add_request(Request {
            id: i as u64,
            prompt: p.clone(),
            params: SamplingParams { max_tokens: o.tokens.len() as u32, ignore_eos: true, temperature: 0.0, ..Default::default() },
        })?;
    }
    let mut got: Vec<(Vec<u32>, Vec<f32>)> = vec![(vec![], vec![]); r.prompts.len()];
    while eng.has_work() {
        for o in eng.step()? {
            let g = &mut got[o.id as usize];
            g.0.extend(&o.new_tokens);
            g.1.extend(&o.logprobs);
        }
    }
    let mut ok = true;
    let mut max_lp = 0f32;
    let mut compared = 0usize;
    let mut ties = 0usize;
    for (i, (o, g)) in r.outputs.iter().zip(&got).enumerate() {
        for j in 0..o.tokens.len() {
            if g.0[j] != o.tokens[j] {
                if o.margins[j] < a.tie_margin {
                    ties += 1;
                } else {
                    ok = false;
                    eprintln!("prompt {i}: token {j} is {} but reference is {} (reference margin {:.3})", g.0[j], o.tokens[j], o.margins[j]);
                }
                break;
            }
            compared += 1;
            max_lp = max_lp.max((g.1[j] - o.logprobs[j]).abs());
        }
    }
    if max_lp > a.logprob_tol {
        ok = false;
        eprintln!("max log-prob difference {max_lp:.4} exceeds {}", a.logprob_tol);
    }
    println!("{}", json!({"pass": ok, "prompts": r.prompts.len(), "tokens_compared": compared, "near_tie_divergences": ties, "max_logprob_diff": max_lp}));
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
