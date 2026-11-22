//! Safetensors shards, memory-mapped. The header format is 8 bytes of little-endian JSON length,
//! the JSON (name → dtype, shape, data_offsets), then the data.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context};
use memmap2::Mmap;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    BF16,
    F16,
    F32,
}

impl DType {
    fn parse(s: &str) -> anyhow::Result<Self> {
        Ok(match s {
            "BF16" => DType::BF16,
            "F16" => DType::F16,
            "F32" => DType::F32,
            other => bail!("unsupported tensor dtype {other}"),
        })
    }
    pub fn size(&self) -> usize {
        match self {
            DType::BF16 | DType::F16 => 2,
            DType::F32 => 4,
        }
    }
}

pub struct TensorView<'a> {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
}

struct Entry {
    shard: usize,
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

pub struct WeightStore {
    shards: Vec<Mmap>,
    index: HashMap<String, Entry>,
}

pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// Round-to-nearest-even f32 → bf16.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((b >> 16) & 1);
    ((b + rounding) >> 16) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = 127 - 15 + 1;
            let mut m = frac;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xff << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

impl WeightStore {
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        let mut files: Vec<_> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        let mut shards = Vec::new();
        let mut index = HashMap::new();
        for (si, path) in files.iter().enumerate() {
            let file = File::open(path)?;
            // SAFETY: the file is opened read-only and not modified while mapped.
            let mm = unsafe { Mmap::map(&file)? };
            let n = u64::from_le_bytes(mm[..8].try_into().unwrap()) as usize;
            let header: Value = serde_json::from_slice(&mm[8..8 + n]).with_context(|| format!("header of {}", path.display()))?;
            let base = 8 + n;
            for (name, meta) in header.as_object().context("header is not an object")? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = DType::parse(meta["dtype"].as_str().context("dtype")?)?;
                let shape = meta["shape"].as_array().context("shape")?.iter().map(|x| x.as_u64().unwrap() as usize).collect();
                let off = meta["data_offsets"].as_array().context("data_offsets")?;
                let (s, e) = (off[0].as_u64().unwrap() as usize, off[1].as_u64().unwrap() as usize);
                index.insert(name.clone(), Entry { shard: si, dtype, shape, start: base + s, end: base + e });
            }
            shards.push(mm);
        }
        Ok(Self { shards, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.index.keys()
    }

    pub fn get(&self, name: &str) -> anyhow::Result<TensorView<'_>> {
        let e = self.index.get(name).with_context(|| format!("weight `{name}` not found"))?;
        Ok(TensorView { dtype: e.dtype, shape: e.shape.clone(), bytes: &self.shards[e.shard][e.start..e.end] })
    }

    /// The tensor as bf16, converting if stored otherwise.
    pub fn bf16(&self, name: &str) -> anyhow::Result<(Vec<u16>, Vec<usize>)> {
        let t = self.get(name)?;
        let v = match t.dtype {
            DType::BF16 => t.bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(),
            DType::F16 => t.bytes.chunks_exact(2).map(|c| f32_to_bf16(f16_to_f32(u16::from_le_bytes([c[0], c[1]])))).collect(),
            DType::F32 => t.bytes.chunks_exact(4).map(|c| f32_to_bf16(f32::from_le_bytes(c.try_into().unwrap()))).collect(),
        };
        Ok((v, t.shape))
    }

    /// The tensor as f32.
    pub fn f32(&self, name: &str) -> anyhow::Result<(Vec<f32>, Vec<usize>)> {
        let t = self.get(name)?;
        let v = match t.dtype {
            DType::BF16 => t.bytes.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            DType::F16 => t.bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            DType::F32 => t.bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        };
        Ok((v, t.shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip() {
        for x in [0.0f32, 1.0, -2.5, 3.140625, 1e-3] {
            let y = bf16_to_f32(f32_to_bf16(x));
            assert!((x - y).abs() <= x.abs() * 0.008, "{x} {y}");
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
    }
}
