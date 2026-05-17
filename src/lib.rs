pub mod context;
pub mod eval;
pub mod fusion;
pub mod tokenize;
pub mod wasm_host;

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub id: String,
    pub score: f32,
    pub snippet: String,
}

pub struct Searcher {
    pub root: PathBuf,
}

impl Searcher {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn search(&self, query: &str, k: usize) -> Vec<SearchHit> {
        let root_str = self.root.to_string_lossy();
        let hits = wasm_host::fusion_search(query, k as u32, &root_str);
        hits.into_iter()
            .map(|h| {
                let snippet = h
                    .payload
                    .get("snippet")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                SearchHit { id: h.id, score: h.score, snippet }
            })
            .collect()
    }
}
