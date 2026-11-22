//! What a model is, as data.
//!
//! A [`spec::ModelSpec`] lists the layers and their shapes. Adapters in [`hf`] build one from a
//! Hugging Face `config.json`; a spec can also be written by hand as JSON, which is how a model is
//! customized (different layer counts, MoE layers, sliding windows) without code. [`weights`]
//! reads safetensors shards and names each tensor the runtime needs.

pub mod hf;
pub mod spec;
pub mod weights;

pub use spec::{AttnSpec, FfnSpec, LayerSpec, ModelSpec, RopeSpec};
pub use weights::{DType, TensorView, WeightStore};
