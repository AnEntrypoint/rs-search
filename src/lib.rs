pub mod bm25;
pub mod context;
pub mod embed;
pub mod embed_cache;
pub mod eval;
pub mod explain;
pub mod fusion;
pub mod git;
pub mod ignore;
pub mod mcp;
pub mod mtime_cache;
#[cfg(feature = "pdf")]
pub mod pdf;
pub mod scanner;
pub mod tokenize;

use std::path::Path;

pub fn run_search(query: &str, root: &Path) -> Vec<bm25::SearchResult> {
    let chunks = scanner::scan_repository(root);
    let results = bm25::search(query, &chunks);
    embed::rerank(results, query, Path::new(""))
}
