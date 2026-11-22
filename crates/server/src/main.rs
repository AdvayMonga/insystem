mod bench;
mod check;
mod engine_thread;
mod http;
mod kbench;
mod text;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use insystem_core::{Engine, EngineConfig};
use insystem_runtime::{CpuExecutor, ExecutorConfig};

#[derive(Parser)]
#[command(name = "insystem", about = "An LLM inference engine.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Serve an OpenAI-compatible HTTP API.
    Serve {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long, default_value = "127.0.0.1:8000")]
        addr: String,
    },
    /// Generate from one prompt and print the text.
    Generate {
        #[command(flatten)]
        engine: EngineArgs,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// Wrap the prompt in the model's chat template as one user message.
        #[arg(long)]
        chat: bool,
    },
    /// Offline throughput on random prompts, run in-process.
    Bench(bench::BenchArgs),
    /// Compare greedy outputs against a reference file written by tools/reference.py.
    Check(check::CheckArgs),
    /// Print the model spec and the compiled forward program.
    Inspect {
        #[arg(long)]
        model: PathBuf,
    },
    /// List kernel variants.
    Kernels,
    /// Time one kernel on synthetic data.
    Kbench(kbench::KbenchArgs),
    /// Measure memory read bandwidth and AMX peak throughput on this machine.
    Peaks,
}

#[derive(Args, Clone, Debug)]
pub struct EngineArgs {
    /// Model directory (config.json or insystem.json, safetensors, tokenizer.json).
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long, default_value_t = 64)]
    pub max_num_seqs: usize,
    #[arg(long, default_value_t = 512)]
    pub max_num_batched_tokens: usize,
    #[arg(long, default_value_t = 4096)]
    pub max_model_len: usize,
    #[arg(long, default_value_t = 3.0)]
    pub kv_cache_gb: f64,
    #[arg(long, default_value_t = 16)]
    pub block_size: usize,
    #[arg(long)]
    pub no_prefix_caching: bool,
    #[arg(long, default_value = insystem_kernels::registry::DEFAULT_MATMUL)]
    pub matmul: String,
    #[arg(long, default_value = insystem_kernels::registry::DEFAULT_ATTENTION)]
    pub attention: String,
    /// Worker threads (0 = one per core).
    #[arg(long, default_value_t = 0)]
    pub threads: usize,
    /// Time every op of the forward pass.
    #[arg(long)]
    pub profile: bool,
}

pub fn build_engine(a: &EngineArgs) -> anyhow::Result<Engine<CpuExecutor>> {
    if a.threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(a.threads).build_global();
    }
    let t0 = std::time::Instant::now();
    let exec = CpuExecutor::load(
        &a.model,
        ExecutorConfig {
            kv_cache_bytes: (a.kv_cache_gb * (1u64 << 30) as f64) as usize,
            block_size: a.block_size,
            max_num_batched_tokens: a.max_num_batched_tokens,
            max_num_seqs: a.max_num_seqs,
            max_model_len: a.max_model_len,
            matmul: a.matmul.clone(),
            attention: a.attention.clone(),
            profile: a.profile,
        },
    )?;
    eprintln!(
        "loaded {} ({:.2}B params, {} layers) in {:.1}s; kernels matmul={} attention={}; KV blocks {} x {}",
        exec.spec.name,
        exec.spec.num_params() as f64 / 1e9,
        exec.spec.layers.len(),
        t0.elapsed().as_secs_f64(),
        exec.kernels.matmul.name,
        exec.kernels.attention.name,
        insystem_core::Executor::cache_spec(&exec).num_blocks,
        a.block_size
    );
    let eos = exec.spec.eos_token_ids.clone();
    Engine::new(
        EngineConfig {
            max_num_seqs: a.max_num_seqs,
            max_num_batched_tokens: a.max_num_batched_tokens,
            max_model_len: a.max_model_len,
            enable_prefix_caching: !a.no_prefix_caching,
            eos_token_ids: eos,
        },
        exec,
    )
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Serve { engine, addr } => http::serve(engine, addr),
        Cmd::Generate { engine, prompt, max_tokens, temperature, chat } => {
            let tok = text::TextCodec::load(&engine.model)?;
            let mut eng = build_engine(&engine)?;
            let prompt_text = if chat { tok.chat_prompt(&[("user".into(), prompt)])? } else { prompt };
            let ids = tok.encode(&prompt_text)?;
            let params = insystem_core::SamplingParams { temperature, max_tokens, ..Default::default() };
            eng.add_request(insystem_core::Request { id: 1, prompt: ids, params })?;
            let t0 = std::time::Instant::now();
            let mut out = Vec::new();
            while eng.has_work() {
                for o in eng.step()? {
                    out.extend(o.new_tokens);
                }
            }
            println!("{}", tok.decode(&out)?);
            eprintln!("{} tokens in {:.2}s", out.len(), t0.elapsed().as_secs_f64());
            Ok(())
        }
        Cmd::Bench(a) => bench::run(a),
        Cmd::Check(a) => check::run(a),
        Cmd::Inspect { model } => {
            let spec = insystem_model::hf::load_spec(&model)?;
            println!("{}", serde_json::to_string_pretty(&spec)?);
            println!("params: {}", spec.num_params());
            print!("{}", insystem_runtime::compile(&spec));
            Ok(())
        }
        Cmd::Kbench(a) => kbench::run(a),
        Cmd::Peaks => {
            let bw = insystem_kernels::peaks::read_bandwidth(2 << 30, 5);
            let amx = insystem_kernels::peaks::amx_tflops(2_000_000);
            println!("{}", serde_json::json!({"threads": rayon::current_num_threads(), "read_gb_s": bw, "amx_bf16_tflops": amx}));
            Ok(())
        }
        Cmd::Kernels => {
            for k in insystem_kernels::registry::MATMUL_KERNELS {
                println!("matmul    {:<14} {}", k.name, k.about);
            }
            for k in insystem_kernels::registry::ATTENTION_KERNELS {
                println!("attention {:<14} {}", k.name, k.about);
            }
            Ok(())
        }
    }
}
