pub mod assemble;
pub mod bm25;
pub mod context;
pub mod embed;
pub mod git;
pub mod ignore;
pub mod mcp;
pub mod mtime_cache;
pub mod scanner;

use std::path::Path;

pub fn run_search(query: &str, root: &Path) -> Vec<bm25::SearchResult> {
    let chunks = scanner::scan_repository(root);
    let results = bm25::search(query, &chunks);
    let models_dir = root.join(".code-search").join("models");
    let model_path = assemble::model_path(&models_dir);
    if model_path.exists() {
        embed::rerank(results, query, &model_path)
    } else {
        results
    }
}
