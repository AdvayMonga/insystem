//! The engine runs on its own thread; HTTP handlers talk to it over channels.

use std::collections::HashMap;
use std::sync::mpsc;

use insystem_core::{Engine, Request, RequestId, RequestOutput};
use insystem_kernels::KernelSet;
use insystem_runtime::CpuExecutor;
use tokio::sync::{mpsc as amp, oneshot};

pub enum Cmd {
    Add { req: Request, tx: amp::UnboundedSender<RequestOutput> },
    Abort(RequestId),
    SwapKernels { matmul: String, attention: String, tol: f32, reply: oneshot::Sender<Result<f32, String>> },
    Stats(oneshot::Sender<serde_json::Value>),
}

pub fn spawn(mut engine: Engine<CpuExecutor>) -> mpsc::Sender<Cmd> {
    let (tx, rx) = mpsc::channel::<Cmd>();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || {
            let mut streams: HashMap<RequestId, amp::UnboundedSender<RequestOutput>> = HashMap::new();
            loop {
                let mut cmds = Vec::new();
                if !engine.has_work() {
                    match rx.recv() {
                        Ok(c) => cmds.push(c),
                        Err(_) => return,
                    }
                }
                while let Ok(c) = rx.try_recv() {
                    cmds.push(c);
                }
                for c in cmds {
                    match c {
                        Cmd::Add { req, tx } => {
                            let id = req.id;
                            match engine.add_request(req) {
                                Ok(()) => {
                                    streams.insert(id, tx);
                                }
                                Err(e) => {
                                    eprintln!("rejected request {id}: {e}");
                                    let _ = tx.send(RequestOutput { id, new_tokens: vec![], logprobs: vec![], finished: Some(insystem_core::FinishReason::Abort), num_prompt_tokens: 0, num_cached_tokens: 0 });
                                }
                            }
                        }
                        Cmd::Abort(id) => engine.abort(id),
                        Cmd::SwapKernels { matmul, attention, tol, reply } => {
                            let r = KernelSet::by_names(&matmul, &attention).and_then(|ks| engine.exec.swap_kernels(ks, tol).map_err(|e| e.to_string()));
                            let _ = reply.send(r);
                        }
                        Cmd::Stats(reply) => {
                            let s = &engine.stats;
                            let _ = reply.send(serde_json::json!({
                                "steps": s.steps,
                                "scheduled_tokens": s.scheduled_tokens,
                                "prefill_tokens": s.prefill_tokens,
                                "sampled_tokens": s.sampled_tokens,
                                "prompt_tokens": s.prompt_tokens,
                                "cached_prompt_tokens": s.cached_prompt_tokens,
                                "exec_seconds": s.exec_time.as_secs_f64(),
                                "sched_seconds": s.sched_time.as_secs_f64(),
                                "running": engine.sched.num_running(),
                                "waiting": engine.sched.num_waiting(),
                                "preemptions": engine.sched.num_preemptions,
                                "kv_free_blocks": engine.sched.kv.num_free(),
                                "kv_cached_blocks": engine.sched.kv.num_cached(),
                                "kernels": {"matmul": engine.exec.kernels.matmul.name, "attention": engine.exec.kernels.attention.name},
                                "profile": engine.exec.profile.as_ref().map(|p| p.to_json()),
                            }));
                        }
                    }
                }
                if !engine.has_work() {
                    continue;
                }
                match engine.step() {
                    Ok(outs) => {
                        for o in outs {
                            let done = o.finished.is_some();
                            let id = o.id;
                            let gone = streams.get(&id).map(|s| s.send(o).is_err()).unwrap_or(true);
                            if done {
                                streams.remove(&id);
                            } else if gone {
                                streams.remove(&id);
                                engine.abort(id);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("engine step failed: {e:#}");
                        std::process::exit(1);
                    }
                }
            }
        })
        .expect("spawn engine thread");
    tx
}
