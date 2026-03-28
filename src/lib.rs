pub mod bm25;
pub mod context;
pub mod ignore;
pub mod mcp;
pub mod mtime_cache;
pub mod scanner;

use std::path::Path;

pub fn run_search(query: &str, root: &Path) -> Vec<bm25::SearchResult> {
    let chunks = scanner::scan_repository(root);
    bm25::search(query, &chunks)
}
