//! Token sampling from one row of logits.

use insystem_core::{Sampled, SamplerSpec};

struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Samples one token. The returned log-probability is under the unmodified model distribution.
pub fn sample(logits: &[f32], spec: &SamplerSpec) -> Sampled {
    let mut mx = f32::NEG_INFINITY;
    let mut arg = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > mx {
            mx = v;
            arg = i;
        }
    }
    let lse = mx + logits.iter().map(|&v| (v - mx).exp()).sum::<f32>().ln();
    if spec.temperature <= 0.0 {
        return Sampled { token: arg as u32, logprob: logits[arg] - lse };
    }
    let inv_t = 1.0 / spec.temperature;
    let mut cand: Vec<(f32, u32)> = logits.iter().enumerate().map(|(i, &v)| ((v - mx) * inv_t, i as u32)).collect();
    let k = if spec.top_k > 0 { (spec.top_k as usize).min(cand.len()) } else { cand.len() };
    if k < cand.len() {
        cand.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
        cand.truncate(k);
    }
    let mut probs: Vec<f32> = cand.iter().map(|c| c.0.exp()).collect();
    let mut sum: f32 = probs.iter().sum();
    if spec.top_p < 1.0 {
        let mut order: Vec<usize> = (0..cand.len()).collect();
        order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        let mut acc = 0.0;
        let mut keep = order.len();
        for (n, &i) in order.iter().enumerate() {
            acc += probs[i] / sum;
            if acc >= spec.top_p {
                keep = n + 1;
                break;
            }
        }
        let kept: Vec<usize> = order[..keep].to_vec();
        cand = kept.iter().map(|&i| cand[i]).collect();
        probs = kept.iter().map(|&i| probs[i]).collect();
        sum = probs.iter().sum();
    }
    let mut r = Rng(spec.rng_seed).next_f32() * sum;
    let mut pick = cand[cand.len() - 1].1;
    for (c, p) in cand.iter().zip(&probs) {
        if r < *p {
            pick = c.1;
            break;
        }
        r -= p;
    }
    Sampled { token: pick, logprob: logits[pick as usize] - lse }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_and_seeded() {
        let logits = vec![0.1, 3.0, 0.2, 2.9];
        let g = sample(&logits, &SamplerSpec { temperature: 0.0, top_k: 0, top_p: 1.0, rng_seed: 1 });
        assert_eq!(g.token, 1);
        let s = SamplerSpec { temperature: 1.0, top_k: 2, top_p: 1.0, rng_seed: 42 };
        let a = sample(&logits, &s);
        assert_eq!(a, sample(&logits, &s));
        assert!(a.token == 1 || a.token == 3);
    }
}
