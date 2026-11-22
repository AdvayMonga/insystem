use serde::{Deserialize, Serialize};

pub type RequestId = u64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    /// 0 means greedy.
    pub temperature: f32,
    /// 0 means no top-k cut.
    pub top_k: u32,
    /// 1.0 means no nucleus cut.
    pub top_p: f32,
    /// Seed for this request's random stream; `None` uses the request id.
    pub seed: Option<u64>,
    pub max_tokens: u32,
    pub stop_token_ids: Vec<u32>,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 0.0, top_k: 0, top_p: 1.0, seed: None, max_tokens: 16, stop_token_ids: vec![], ignore_eos: false }
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    pub id: RequestId,
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    Abort,
}

impl FinishReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Abort => "abort",
        }
    }
}

/// What one engine step produced for one request.
#[derive(Clone, Debug)]
pub struct RequestOutput {
    pub id: RequestId,
    pub new_tokens: Vec<u32>,
    pub logprobs: Vec<f32>,
    pub finished: Option<FinishReason>,
    pub num_prompt_tokens: usize,
    /// Prompt tokens served from the prefix cache at admission.
    pub num_cached_tokens: usize,
}
