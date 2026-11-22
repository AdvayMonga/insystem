//! Per-op wall time, grouped by [`crate::Instr::label`].

use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct Profile {
    pub ops: BTreeMap<String, (Duration, u64)>,
    pub steps: u64,
    pub tokens: u64,
}

impl Profile {
    pub fn add(&mut self, label: String, d: Duration) {
        let e = self.ops.entry(label).or_default();
        e.0 += d;
        e.1 += 1;
    }

    pub fn total(&self) -> Duration {
        self.ops.values().map(|v| v.0).sum()
    }

    pub fn report(&self) -> String {
        let total = self.total().as_secs_f64().max(1e-9);
        let mut rows: Vec<_> = self.ops.iter().collect();
        rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
        let mut s = format!("steps {}  tokens {}  op time {:.3}s\n", self.steps, self.tokens, total);
        for (k, (d, n)) in rows {
            s += &format!("  {:<18} {:>9.3}s {:>6.1}%  calls {}\n", k, d.as_secs_f64(), 100.0 * d.as_secs_f64() / total, n);
        }
        s
    }

    pub fn to_json(&self) -> serde_json::Value {
        let ops: serde_json::Map<String, serde_json::Value> =
            self.ops.iter().map(|(k, (d, n))| (k.clone(), serde_json::json!({"seconds": d.as_secs_f64(), "calls": n}))).collect();
        serde_json::json!({"steps": self.steps, "tokens": self.tokens, "ops": ops})
    }
}
