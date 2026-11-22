//! The CPU executor.
//!
//! [`program::compile`] turns a [`insystem_model::ModelSpec`] into a [`program::Program`]: the
//! flat list of ops one forward pass runs, over named scratch buffers. [`weights::ModelWeights`]
//! holds the weights packed for the selected kernels. [`executor::CpuExecutor`] runs the program
//! for each [`insystem_core::BatchPlan`], samples, and can swap kernels on the live engine after
//! checking the new ones against the current ones on a probe batch.

pub mod executor;
pub mod profile;
pub mod program;
pub mod sampler;
pub mod weights;

pub use executor::{CpuExecutor, ExecutorConfig};
pub use program::{compile, Instr, Program};
