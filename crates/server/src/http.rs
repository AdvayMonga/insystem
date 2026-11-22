//! OpenAI-compatible HTTP API: `/v1/completions`, `/v1/chat/completions` (streaming and not),
//! `/v1/models`, `/health`, `/metrics`, and `/admin/kernels` to switch kernels on the live engine.

use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use insystem_core::{Request, RequestOutput, SamplingParams};
use serde_json::{json, Value};
use tokio::sync::{mpsc as amp, oneshot};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::engine_thread::{self, Cmd};
use crate::text::TextCodec;
use crate::EngineArgs;

struct App {
    tx: mpsc::Sender<Cmd>,
    codec: Option<TextCodec>,
    next_id: AtomicU64,
    model: String,
}

type Shared = Arc<App>;

fn err(code: StatusCode, msg: impl ToString) -> Response {
    (code, Json(json!({"error": {"message": msg.to_string()}}))).into_response()
}

pub fn serve(args: EngineArgs, addr: String) -> anyhow::Result<()> {
    let codec = match TextCodec::load(&args.model) {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("no tokenizer ({e}); prompts must be token id arrays");
            None
        }
    };
    let model = args.model.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "model".into());
    let engine = crate::build_engine(&args)?;
    let tx = engine_thread::spawn(engine);
    let app = Arc::new(App { tx, codec, next_id: AtomicU64::new(1), model });
    let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?;
    rt.block_on(async move {
        let router = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route("/v1/models", get(models))
            .route("/v1/completions", post(completions))
            .route("/v1/chat/completions", post(chat))
            .route("/metrics", get(metrics))
            .route("/admin/kernels", post(swap_kernels))
            .with_state(app);
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        eprintln!("serving on http://{addr}");
        axum::serve(listener, router).await?;
        Ok(())
    })
}

async fn models(State(app): State<Shared>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": app.model, "object": "model", "owned_by": "insystem"}]}))
}

async fn metrics(State(app): State<Shared>) -> Response {
    let (tx, rx) = oneshot::channel();
    if app.tx.send(Cmd::Stats(tx)).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match rx.await {
        Ok(v) => Json(v).into_response(),
        Err(_) => err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

async fn swap_kernels(State(app): State<Shared>, Json(body): Json<Value>) -> Response {
    let get = |k: &str| body.get(k).and_then(Value::as_str).map(String::from);
    let (Some(matmul), Some(attention)) = (get("matmul"), get("attention")) else {
        return err(StatusCode::BAD_REQUEST, "need `matmul` and `attention`");
    };
    let tol = body.get("tolerance").and_then(Value::as_f64).unwrap_or(1e-2) as f32;
    let (tx, rx) = oneshot::channel();
    if app.tx.send(Cmd::SwapKernels { matmul, attention, tol, reply: tx }).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    match rx.await {
        Ok(Ok(e)) => Json(json!({"ok": true, "probe_rel_error": e})).into_response(),
        Ok(Err(e)) => err(StatusCode::UNPROCESSABLE_ENTITY, e),
        Err(_) => err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped"),
    }
}

fn sampling_params(b: &Value) -> SamplingParams {
    let f = |k: &str, d: f64| b.get(k).and_then(Value::as_f64).unwrap_or(d) as f32;
    SamplingParams {
        temperature: f("temperature", 1.0),
        top_p: f("top_p", 1.0),
        top_k: b.get("top_k").and_then(Value::as_i64).filter(|&k| k > 0).unwrap_or(0) as u32,
        seed: b.get("seed").and_then(Value::as_u64),
        max_tokens: b.get("max_tokens").or_else(|| b.get("max_completion_tokens")).and_then(Value::as_u64).unwrap_or(16) as u32,
        stop_token_ids: b.get("stop_token_ids").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect()).unwrap_or_default(),
        ignore_eos: b.get("ignore_eos").and_then(Value::as_bool).unwrap_or(false),
    }
}

/// Turns streamed token ids into text deltas, holding back incomplete UTF-8.
struct Detok {
    ids: Vec<u32>,
    sent: usize,
}

impl Detok {
    fn push(&mut self, codec: Option<&TextCodec>, toks: &[u32]) -> String {
        self.ids.extend_from_slice(toks);
        let Some(c) = codec else { return String::new() };
        let full = c.decode(&self.ids).unwrap_or_default();
        if full.ends_with('\u{FFFD}') || full.len() < self.sent || !full.is_char_boundary(self.sent) {
            return String::new();
        }
        let d = full[self.sent..].to_string();
        self.sent = full.len();
        d
    }
}

async fn completions(State(app): State<Shared>, Json(body): Json<Value>) -> Response {
    let prompt = match body.get("prompt") {
        Some(Value::String(s)) => match &app.codec {
            Some(c) => match c.encode(s) {
                Ok(ids) => ids,
                Err(e) => return err(StatusCode::BAD_REQUEST, e),
            },
            None => return err(StatusCode::BAD_REQUEST, "no tokenizer loaded; send token ids"),
        },
        Some(Value::Array(a)) if a.iter().all(Value::is_u64) => a.iter().map(|x| x.as_u64().unwrap() as u32).collect(),
        Some(Value::Array(a)) if a.len() == 1 => match (&a[0], &app.codec) {
            (Value::String(s), Some(c)) => c.encode(s).unwrap_or_default(),
            (Value::Array(ids), _) => ids.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
            _ => return err(StatusCode::BAD_REQUEST, "unsupported prompt"),
        },
        _ => return err(StatusCode::BAD_REQUEST, "missing prompt"),
    };
    run(app, body, prompt, false).await
}

async fn chat(State(app): State<Shared>, Json(body): Json<Value>) -> Response {
    let Some(codec) = &app.codec else { return err(StatusCode::BAD_REQUEST, "no tokenizer loaded") };
    let Some(msgs) = body.get("messages").and_then(Value::as_array) else { return err(StatusCode::BAD_REQUEST, "missing messages") };
    let msgs: Vec<(String, String)> = msgs
        .iter()
        .map(|m| (m.get("role").and_then(Value::as_str).unwrap_or("user").to_string(), m.get("content").and_then(Value::as_str).unwrap_or("").to_string()))
        .collect();
    let prompt = match codec.chat_prompt(&msgs).and_then(|p| codec.encode(&p)) {
        Ok(p) => p,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    run(app, body, prompt, true).await
}

async fn run(app: Shared, body: Value, prompt: Vec<u32>, is_chat: bool) -> Response {
    let id = app.next_id.fetch_add(1, Ordering::Relaxed);
    let params = sampling_params(&body);
    let n_prompt = prompt.len();
    let (tx, mut rx) = amp::unbounded_channel::<RequestOutput>();
    if app.tx.send(Cmd::Add { req: Request { id, prompt, params }, tx }).is_err() {
        return err(StatusCode::SERVICE_UNAVAILABLE, "engine stopped");
    }
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let rid = format!("{}-{id}", if is_chat { "chatcmpl" } else { "cmpl" });
    let model = app.model.clone();
    let object = if is_chat { "chat.completion" } else { "text_completion" };
    if !stream {
        let mut det = Detok { ids: vec![], sent: 0 };
        let mut text = String::new();
        let mut finish = None;
        let mut cached = 0;
        while let Some(o) = rx.recv().await {
            text += &det.push(app.codec.as_ref(), &o.new_tokens);
            cached += o.num_cached_tokens;
            if o.finished.is_some() {
                finish = o.finished;
                break;
            }
        }
        let choice = if is_chat {
            json!({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish.map(|f| f.as_str())})
        } else {
            json!({"index": 0, "text": text, "token_ids": det.ids, "finish_reason": finish.map(|f| f.as_str())})
        };
        return Json(json!({
            "id": rid, "object": object, "model": model, "choices": [choice],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": det.ids.len(), "total_tokens": n_prompt + det.ids.len(),
                      "prompt_tokens_details": {"cached_tokens": cached}},
        }))
        .into_response();
    }
    let include_usage = body.pointer("/stream_options/include_usage").and_then(Value::as_bool).unwrap_or(false);
    let (etx, erx) = amp::unbounded_channel::<Event>();
    let chunk_obj = if is_chat { "chat.completion.chunk" } else { "text_completion" };
    let codec_app = app.clone();
    tokio::spawn(async move {
        let mut det = Detok { ids: vec![], sent: 0 };
        while let Some(o) = rx.recv().await {
            let delta = det.push(codec_app.codec.as_ref(), &o.new_tokens);
            let fin = o.finished.map(|f| f.as_str());
            let choice = if is_chat {
                json!({"index": 0, "delta": {"content": delta}, "finish_reason": fin})
            } else {
                json!({"index": 0, "text": delta, "token_ids": o.new_tokens, "finish_reason": fin})
            };
            let ev = json!({"id": rid, "object": chunk_obj, "model": model, "choices": [choice]});
            if etx.send(Event::default().data(ev.to_string())).is_err() {
                let _ = codec_app.tx.send(Cmd::Abort(id));
                return;
            }
            if o.finished.is_some() {
                if include_usage {
                    let u = json!({"id": rid, "object": chunk_obj, "model": model, "choices": [],
                        "usage": {"prompt_tokens": n_prompt, "completion_tokens": det.ids.len(), "total_tokens": n_prompt + det.ids.len()}});
                    let _ = etx.send(Event::default().data(u.to_string()));
                }
                let _ = etx.send(Event::default().data("[DONE]"));
                return;
            }
        }
    });
    Sse::new(UnboundedReceiverStream::new(erx).map(Ok::<Event, Infallible>)).into_response()
}
