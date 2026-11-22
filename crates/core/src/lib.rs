//! Model-independent engine core.
//!
//! The core owns every request, the scheduler and the KV block allocator. Each step it hands the
//! executor one read-only [`plan::BatchPlan`] and gets back one sampled token per sampling row.
//! It never touches weights, tensors or kernels.

pub mod engine;
pub mod kv;
pub mod plan;
pub mod request;
pub mod scheduler;

pub use engine::{Engine, EngineConfig, EngineStats};
pub use plan::{BatchPlan, CacheSpec, Executor, Sampled, SamplerSpec, SeqSlice, StepOutput};
pub use request::{FinishReason, Request, RequestId, RequestOutput, SamplingParams};
