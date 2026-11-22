//! The forward pass as a list of ops.
//!
//! `compile` reads the spec and emits, per layer: attention norm, fused QKV projection, q/k norm,
//! RoPE, KV write, attention, output projection, residual add fused into the FFN norm, then the
//! FFN (fused gate/up projection, SiLU-mul, down projection, or a routed MoE). The residual add of
//! each layer's output is fused into the next layer's first norm. After the last layer only the
//! rows that are sampled are gathered, normed and projected to logits.

use std::fmt;

use insystem_model::{FfnSpec, ModelSpec};

/// Scratch buffers, each sized for the step's token count (or sampled-row count).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Buf {
    /// Residual stream `[T, hidden]`.
    Resid,
    /// Normed hidden `[T, hidden]`.
    Normed,
    /// `[T, (n_heads + 2 n_kv_heads) head_dim]`.
    Qkv,
    /// `[T, n_heads head_dim]`.
    Attn,
    /// Projection output added to the residual `[T, hidden]`.
    Delta,
    /// `[T, 2 intermediate]`.
    GateUp,
    /// `[T, intermediate]`.
    Act,
    /// Sampled rows `[S, hidden]`.
    Last,
    /// `[S, vocab]`.
    Logits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lin {
    Qkv,
    O,
    GateUp,
    Down,
    LmHead,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Instr {
    Embed,
    RmsNorm { layer: usize },
    /// `Resid += Delta; Normed = norm(Resid)` with the norm of `(layer, which)`.
    AddRmsNorm { layer: usize, ffn: bool },
    Linear { layer: usize, lin: Lin, src: Buf, dst: Buf },
    QkNormRope { layer: usize },
    KvWrite { layer: usize },
    Attention { layer: usize },
    SiluMul { inter: usize },
    Moe { layer: usize },
    /// `Last[s] = Resid[row] + Delta[row]` for each sampled row.
    GatherLast,
    FinalNorm,
    Sample,
}

impl Instr {
    /// Label used to group time in profiles.
    pub fn label(&self) -> String {
        match self {
            Instr::Linear { lin, .. } => format!("linear.{lin:?}").to_lowercase(),
            Instr::AddRmsNorm { .. } | Instr::RmsNorm { .. } | Instr::FinalNorm => "norm".into(),
            other => format!("{other:?}").split([' ', '{']).next().unwrap().to_lowercase(),
        }
    }
}

pub struct Program {
    pub instrs: Vec<Instr>,
}

pub fn compile(spec: &ModelSpec) -> Program {
    let mut v = vec![Instr::Embed];
    for (i, l) in spec.layers.iter().enumerate() {
        if i == 0 {
            v.push(Instr::RmsNorm { layer: 0 });
        } else {
            v.push(Instr::AddRmsNorm { layer: i, ffn: false });
        }
        v.push(Instr::Linear { layer: i, lin: Lin::Qkv, src: Buf::Normed, dst: Buf::Qkv });
        v.push(Instr::QkNormRope { layer: i });
        v.push(Instr::KvWrite { layer: i });
        v.push(Instr::Attention { layer: i });
        v.push(Instr::Linear { layer: i, lin: Lin::O, src: Buf::Attn, dst: Buf::Delta });
        v.push(Instr::AddRmsNorm { layer: i, ffn: true });
        match l.ffn {
            FfnSpec::Dense { intermediate } => {
                v.push(Instr::Linear { layer: i, lin: Lin::GateUp, src: Buf::Normed, dst: Buf::GateUp });
                v.push(Instr::SiluMul { inter: intermediate });
                v.push(Instr::Linear { layer: i, lin: Lin::Down, src: Buf::Act, dst: Buf::Delta });
            }
            FfnSpec::Moe { .. } => v.push(Instr::Moe { layer: i }),
        }
    }
    v.push(Instr::GatherLast);
    v.push(Instr::FinalNorm);
    v.push(Instr::Linear { layer: 0, lin: Lin::LmHead, src: Buf::Last, dst: Buf::Logits });
    v.push(Instr::Sample);
    Program { instrs: v }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, ins) in self.instrs.iter().enumerate() {
            writeln!(f, "{i:4}  {ins:?}")?;
        }
        Ok(())
    }
}
