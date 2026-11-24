//! Zero-dependency recursive-descent JSON parser and serializer.
//! Powers unified `JobSpec`, checkpoints, `quality_report.json`, Web UI API, and `--ndjson` events.

use std::collections::BTreeMap;

use super::error::{PipelineError, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| n.round() as i64)
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().and_then(|n| if n >= 0.0 { Some(n.round() as u64) } else { None })
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object().and_then(|m| m.get(key))
    }

    pub fn to_compact_string(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    pub fn to_pretty_string(&self) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Number(n) => {
                if !n.is_finite() {
                    out.push_str("null");
                } else if (*n - n.trunc()).abs() < 1e-12 && n.abs() < 9e15 {
                    out.push_str(&format!("{}", *n as i64));
                } else {
                    out.push_str(&format!("{}", n));
                }
            }
            Self::String(s) => write_escaped_str(s, out),
            Self::Array(arr) => {
                out.push('[');
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    v.write_compact(out);
                }
                out.push(']');
            }
            Self::Object(map) => {
                out.push('{');
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped_str(k, out);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, indent: usize) {
        match self {
            Self::Array(arr) if !arr.is_empty() => {
                out.push_str("[\n");
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    push_indent(out, indent + 2);
                    v.write_pretty(out, indent + 2);
                }
                out.push('\n');
                push_indent(out, indent);
                out.push(']');
            }
            Self::Object(map) if !map.is_empty() => {
                out.push_str("{\n");
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    push_indent(out, indent + 2);
                    write_escaped_str(k, out);
                    out.push_str(": ");
                    v.write_pretty(out, indent + 2);
                }
                out.push('\n');
                push_indent(out, indent);
                out.push('}');
            }
            _ => self.write_compact(out),
        }
    }
}

fn push_indent(out: &mut String, n: usize) {
    for _ in 0..n {
        out.push(' ');
    }
}

pub fn write_escaped_str(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn parse_json(input: &str) -> Result<JsonValue> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    p.skip_ws();
    let val = p.parse_value(0)?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return Err(PipelineError::Format(format!(
            "Unexpected trailing JSON bytes at offset {}",
            p.pos
        )));
    }
    Ok(val)
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue> {
        if depth > 64 {
            return Err(PipelineError::Format("JSON nesting too deep".into()));
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(depth + 1),
            Some(b'[') => self.parse_array(depth + 1),
            Some(b'"') => self.parse_string().map(JsonValue::String),
            Some(b't') => self.expect_literal(b"true", JsonValue::Bool(true)),
            Some(b'f') => self.expect_literal(b"false", JsonValue::Bool(false)),
            Some(b'n') => self.expect_literal(b"null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            other => Err(PipelineError::Format(format!(
                "Unexpected JSON token {:?} at byte {}",
                other.map(|c| c as char),
                self.pos
            ))),
        }
    }

    fn expect_literal(&mut self, lit: &[u8], val: JsonValue) -> Result<JsonValue> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(PipelineError::Format(format!("Invalid JSON literal at byte {}", self.pos)))
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        if self.peek() != Some(b'"') {
            return Err(PipelineError::Format("Expected '\"'".into()));
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            if b == b'"' {
                return Ok(out);
            }
            if b == b'\\' {
                let esc = *self
                    .bytes
                    .get(self.pos)
                    .ok_or_else(|| PipelineError::Format("Unterminated escape".into()))?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'u' => {
                        if self.pos + 4 > self.bytes.len() {
                            return Err(PipelineError::Format("Truncated \\u escape".into()));
                        }
                        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
                            .map_err(|_| PipelineError::Format("Invalid UTF-8 in \\u".into()))?;
                        let code = u32::from_str_radix(hex, 16)
                            .map_err(|_| PipelineError::Format("Invalid hex in \\u".into()))?;
                        self.pos += 4;
                        if let Some(c) = char::from_u32(code) {
                            out.push(c);
                        }
                    }
                    _ => return Err(PipelineError::Format("Unsupported JSON escape".into())),
                }
            } else {
                // Handle UTF-8 multi-byte characters safely
                let start = self.pos - 1;
                if b < 0x80 {
                    out.push(b as char);
                } else {
                    let s = std::str::from_utf8(&self.bytes[start..])
                        .map_err(|_| PipelineError::Format("Invalid UTF-8 in JSON string".into()))?;
                    if let Some(ch) = s.chars().next() {
                        out.push(ch);
                        self.pos = start + ch.len_utf8();
                    }
                }
            }
        }
        Err(PipelineError::Format("Unterminated JSON string".into()))
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        while let Some(b'0'..=b'9') = self.peek() {
            self.pos += 1;
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| PipelineError::Format("Invalid number bytes".into()))?;
        let n = s
            .parse::<f64>()
            .map_err(|_| PipelineError::Format(format!("Invalid JSON number '{}'", s)))?;
        Ok(JsonValue::Number(n))
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue> {
        self.pos += 1; // '['
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(arr));
        }
        loop {
            arr.push(self.parse_value(depth)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(arr));
                }
                _ => return Err(PipelineError::Format(format!("Expected ',' or ']' at byte {}", self.pos))),
            }
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue> {
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(PipelineError::Format(format!("Expected ':' at byte {}", self.pos)));
            }
            self.pos += 1;
            let val = self.parse_value(depth)?;
            map.insert(key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                _ => return Err(PipelineError::Format(format!("Expected ',' or '}}' at byte {}", self.pos))),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_roundtrip() {
        let raw = r#"{"name":"stereo_job","width":192,"ratio":0.75,"flags":[true,false,null],"nested":{"msg":"hello \"world\""}}"#;
        let parsed = parse_json(raw).unwrap();
        assert_eq!(parsed.get("name").and_then(|v| v.as_str()), Some("stereo_job"));
        assert_eq!(parsed.get("width").and_then(|v| v.as_u64()), Some(192));
        let reparsed = parse_json(&parsed.to_pretty_string()).unwrap();
        assert_eq!(parsed, reparsed);
    }
}
