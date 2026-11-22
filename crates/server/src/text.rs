//! Tokenizer and chat template.

use std::path::Path;

use anyhow::Context;
use tokenizers::Tokenizer;

pub struct TextCodec {
    tok: Tokenizer,
    chat_template: Option<String>,
    bos: Option<String>,
    eos: Option<String>,
}

fn token_str(v: Option<&serde_json::Value>) -> Option<String> {
    match v? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o.get("content").and_then(|c| c.as_str()).map(String::from),
        _ => None,
    }
}

impl TextCodec {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("tokenizer.json: {e}"))?;
        let cfg: Option<serde_json::Value> = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok().and_then(|s| serde_json::from_str(&s).ok());
        let chat_template = cfg.as_ref().and_then(|c| c.get("chat_template")).and_then(|v| v.as_str()).map(String::from);
        Ok(Self {
            tok,
            chat_template,
            bos: token_str(cfg.as_ref().and_then(|c| c.get("bos_token"))),
            eos: token_str(cfg.as_ref().and_then(|c| c.get("eos_token"))),
        })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(|e| anyhow::anyhow!("{e}"))?.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tok.decode(ids, true).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Renders `messages` with the model's Jinja chat template, ending with the assistant turn.
    pub fn chat_prompt(&self, messages: &[(String, String)]) -> anyhow::Result<String> {
        let tpl = self.chat_template.as_deref().context("the model has no chat_template")?;
        let mut env = minijinja::Environment::new();
        minijinja_contrib_pycompat(&mut env);
        env.add_template("chat", tpl)?;
        let msgs: Vec<_> = messages.iter().map(|(r, c)| minijinja::context! { role => r, content => c }).collect();
        let out = env.get_template("chat")?.render(minijinja::context! {
            messages => msgs,
            add_generation_prompt => true,
            enable_thinking => false,
            bos_token => self.bos.clone().unwrap_or_default(),
            eos_token => self.eos.clone().unwrap_or_default(),
        })?;
        Ok(out)
    }
}

/// Python string methods that Hugging Face chat templates call.
fn minijinja_contrib_pycompat(env: &mut minijinja::Environment) {
    use minijinja::value::Value;
    use minijinja::{Error, ErrorKind, State};
    env.set_unknown_method_callback(|_state: &State, value: &Value, method: &str, args: &[Value]| -> Result<Value, Error> {
        let Some(s) = value.as_str() else {
            return Err(Error::new(ErrorKind::UnknownMethod, format!("{method} on non-string")));
        };
        let arg = |i: usize| args.get(i).and_then(|v| v.as_str()).map(String::from);
        Ok(match method {
            "startswith" => Value::from(arg(0).is_some_and(|a| s.starts_with(&a))),
            "endswith" => Value::from(arg(0).is_some_and(|a| s.ends_with(&a))),
            "strip" => Value::from(match arg(0) { Some(c) => s.trim_matches(|x| c.contains(x)).to_string(), None => s.trim().to_string() }),
            "lstrip" => Value::from(match arg(0) { Some(c) => s.trim_start_matches(|x| c.contains(x)).to_string(), None => s.trim_start().to_string() }),
            "rstrip" => Value::from(match arg(0) { Some(c) => s.trim_end_matches(|x| c.contains(x)).to_string(), None => s.trim_end().to_string() }),
            "split" => {
                let parts: Vec<Value> = match arg(0) { Some(sep) => s.split(sep.as_str()).map(Value::from).collect(), None => s.split_whitespace().map(Value::from).collect() };
                Value::from(parts)
            }
            "replace" => Value::from(s.replace(&arg(0).unwrap_or_default(), &arg(1).unwrap_or_default())),
            "upper" => Value::from(s.to_uppercase()),
            "lower" => Value::from(s.to_lowercase()),
            _ => return Err(Error::new(ErrorKind::UnknownMethod, format!("string method {method}"))),
        })
    });
}
