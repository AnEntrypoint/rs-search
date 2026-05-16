pub mod bm25;
pub mod context;
pub mod embed;
#[cfg(feature = "vector")]
pub mod embed_cache;
pub mod eval;
pub mod explain;
pub mod fusion;
pub mod git;
pub mod ignore;
#[cfg(not(target_arch = "wasm32"))]
pub mod mcp;
pub mod mtime_cache;
pub mod resolve_index;
#[cfg(feature = "pdf")]
pub mod pdf;
pub mod scanner;
pub mod tokenize;
#[cfg(target_arch = "wasm32")]
pub mod wasm_host;

use std::path::{Path, PathBuf};

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

    #[cfg(not(target_arch = "wasm32"))]
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchHit> {
        let results = run_search(query, &self.root);
        results
            .into_iter()
            .take(k)
            .map(|r| SearchHit {
                id: format!("{}:{}", r.chunk.file_path, r.chunk.line_start),
                score: r.score as f32,
                snippet: r.chunk.content,
            })
            .collect()
    }

    #[cfg(target_arch = "wasm32")]
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

#[cfg(not(target_arch = "wasm32"))]
pub fn run_search(query: &str, root: &Path) -> Vec<bm25::SearchResult> {
    let chunks = scanner::scan_repository(root);
    let results = bm25::search(query, &chunks);
    embed::rerank(results, query, Path::new(""))
}
