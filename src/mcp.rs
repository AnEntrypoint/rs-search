use std::io::{self, BufRead, Write};
use std::path::Path;
use std::collections::HashMap;
use serde_json::{json, Value};
use crate::scanner::scan_repository;
use crate::bm25::{search, search_texts};
use crate::embed::{rerank, vector_search_texts};
use crate::git::{scan_git_commits, commits_to_searchable};
use crate::assemble;
use crate::context::{find_enclosing_context, get_file_total_lines};

struct IndexCache {
    chunks: Vec<crate::scanner::Chunk>,
    indexed_at: u64,
}

fn dir_mtime(path: &Path) -> u64 {
    std::fs::metadata(path).ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn run_mcp_server() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut cache: HashMap<String, IndexCache> = HashMap::new();

    for line in stdin.lock().lines() {
        let line = match line { Ok(l) => l, Err(_) => break };
        if line.trim().is_empty() { continue; }
        let msg: Value = match serde_json::from_str(&line) { Ok(v) => v, Err(_) => continue };

        let id = msg.get("id").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(|v| v.as_str()).unwrap_or("");

        let response = match method {
            "initialize" => json!({
                "jsonrpc": "2.0", "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": { "tools": {} },
                    "serverInfo": { "name": "rs-search", "version": "0.1.0" }
                }
            }),
            "tools/list" => json!({
                "jsonrpc": "2.0", "id": id,
                "result": { "tools": [{
                    "name": "search",
                    "description": "Search a code repository. Returns 4 sections: BM25 code results, vector code results, most relevant commits (BM25), most relevant commits (vector).",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "repository_path": { "type": "string", "description": "Path to repository (defaults to current directory)" },
                            "query": { "type": "string", "description": "Natural language search query" }
                        },
                        "required": ["query"]
                    }
                }]}
            }),
            "tools/call" => {
                let params = msg.get("params").cloned().unwrap_or(json!({}));
                let tool = params.get("name").and_then(|v| v.as_str()).unwrap_or("");
                if tool != "search" {
                    err_response(id, format!("Unknown tool: {}", tool))
                } else {
                    let args = params.get("arguments").cloned().unwrap_or(json!({}));
                    let query = args.get("query").and_then(|v| v.as_str()).unwrap_or("");
                    if query.is_empty() {
                        err_response(id, "query is required".into())
                    } else {
                        let repo = args.get("repository_path").and_then(|v| v.as_str())
                            .map(String::from)
                            .unwrap_or_else(|| std::env::current_dir().unwrap().to_string_lossy().to_string());
                        let repo_path = Path::new(&repo);
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
                        let dir_mt = dir_mtime(repo_path);
                        let cached = cache.get(&repo).filter(|c| c.indexed_at >= dir_mt);
                        let chunks = if let Some(c) = cached {
                            c.chunks.clone()
                        } else {
                            let ch = scan_repository(repo_path);
                            cache.insert(repo.clone(), IndexCache { chunks: ch.clone(), indexed_at: now });
                            ch
                        };
                        let models_dir = repo_path.join(".code-search").join("models");
                        let model_path = assemble::model_path(&models_dir);
                        let model_exists = model_path.exists();

                        let bm25_results = search(query, &chunks);
                        let vector_results = if model_exists {
                            rerank(bm25_results.clone(), query, &model_path)
                        } else { bm25_results.clone() };

                        let commits = scan_git_commits(repo_path, 200);
                        let commit_texts = commits_to_searchable(&commits);
                        let bm25_commits = search_texts(query, &commit_texts);
                        let vector_commits = if model_exists {
                            vector_search_texts(query, &commit_texts, &model_path)
                        } else { vec![] };

                        let text = format_all(query, repo_path, &bm25_results, &vector_results, &bm25_commits, &vector_commits);
                        json!({ "jsonrpc": "2.0", "id": id, "result": { "content": [{ "type": "text", "text": text }] } })
                    }
                }
            },
            "notifications/initialized" | "notifications/cancelled" => continue,
            _ => json!({ "jsonrpc": "2.0", "id": id, "error": { "code": -32601, "message": "Method not found" } }),
        };

        let _ = writeln!(out, "{}", response);
        let _ = out.flush();
    }
}

fn err_response(id: Value, msg: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": { "content": [{ "type": "text", "text": msg }], "isError": true } })
}

fn format_code_results(results: &[crate::bm25::SearchResult], root: &Path) -> String {
    if results.is_empty() { return "  (no results)\n".into(); }
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        let total = get_file_total_lines(root, &r.chunk.file_path).map(|n| format!(" [{}L]", n)).unwrap_or_default();
        let ctx = find_enclosing_context(&r.chunk.content, r.chunk.line_start).map(|c| format!(" (in: {})", c)).unwrap_or_default();
        let score_pct = (r.score * 100.0).round() as u64;
        out.push_str(&format!("{}. {}{}: {}-{}{} (score: {}%)\n",
            i + 1, r.chunk.file_path, total, r.chunk.line_start, r.chunk.line_end, ctx, score_pct));
        if let Some(vs) = r.vector_score {
            out.push_str(&format!("   BM25: {:.2}  Vector: {:.4}\n", r.bm25_raw, vs));
        } else {
            out.push_str(&format!("   BM25: {:.2}\n", r.bm25_raw));
        }
        for line in r.chunk.content.split('\n').take(5) {
            out.push_str(&format!("   {}\n", line));
        }
        out.push('\n');
    }
    out
}

fn format_commit_hashes(items: &[(String, f64)]) -> String {
    if items.is_empty() { return "  (no results)\n".into(); }
    items.iter().take(10).enumerate()
        .map(|(i, (hash, score))| format!("{}. {} (score: {:.0}%)\n", i + 1, &hash[..hash.len().min(12)], score * 100.0))
        .collect()
}

fn format_commit_hashes_vec(items: &[(String, f32)]) -> String {
    if items.is_empty() { return "  (no results)\n".into(); }
    items.iter().take(10).enumerate()
        .map(|(i, (hash, score))| format!("{}. {} (score: {:.0}%)\n", i + 1, &hash[..hash.len().min(12)], ((*score + 1.0) / 2.0) * 100.0))
        .collect()
}

fn format_all(
    query: &str,
    root: &Path,
    bm25: &[crate::bm25::SearchResult],
    vector: &[crate::bm25::SearchResult],
    bm25_commits: &[(String, f64)],
    vec_commits: &[(String, f32)],
) -> String {
    let mut out = format!("Search results for: \"{}\"\n\n", query);
    out.push_str("=== BM25 RESULTS ===\n");
    out.push_str(&format_code_results(bm25, root));
    out.push_str("=== VECTOR RESULTS ===\n");
    out.push_str(&format_code_results(vector, root));
    out.push_str("=== MOST RELEVANT COMMITS (BM25) ===\n");
    out.push_str(&format_commit_hashes(bm25_commits));
    out.push_str("=== MOST RELEVANT COMMITS (vector) ===\n");
    out.push_str(&format_commit_hashes_vec(vec_commits));
    out
}
