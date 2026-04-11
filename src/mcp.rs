use std::io::{self, BufRead, Write};
use std::path::Path;
use std::collections::HashMap;
use serde_json::{json, Value};
use crate::scanner::scan_repository;
use crate::bm25::search;
use crate::embed::rerank;
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
                    "description": "Search through a code repository using BM25 + vector hybrid search.",
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
                        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
                        let dir_mt = dir_mtime(repo_path);
                        let cached = cache.get(&repo).filter(|c| c.indexed_at >= dir_mt);
                        let chunks = if let Some(c) = cached {
                            c.chunks.clone()
                        } else {
                            let ch = scan_repository(repo_path);
                            cache.insert(repo.clone(), IndexCache { chunks: ch.clone(), indexed_at: now });
                            ch
                        };
                        let bm25_results = search(query, &chunks);
                        let models_dir = repo_path.join(".code-search").join("models");
                        let model_path = assemble::model_path(&models_dir);
                        let results = if model_path.exists() {
                            rerank(bm25_results, query, &model_path)
                        } else {
                            bm25_results
                        };
                        let text = format_results(&results, query, repo_path);
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

fn format_results(results: &[crate::bm25::SearchResult], query: &str, root: &Path) -> String {
    if results.is_empty() { return format!("No results found for: \"{}\"", query); }
    let plural = if results.len() != 1 { "s" } else { "" };
    let mut out = format!("Found {} result{} for: \"{}\"\n\n", results.len(), plural, query);
    for (i, r) in results.iter().enumerate() {
        let total = get_file_total_lines(root, &r.chunk.file_path).map(|n| format!(" [{}L]", n)).unwrap_or_default();
        let ctx = find_enclosing_context(&r.chunk.content, r.chunk.line_start).map(|c| format!(" (in: {})", c)).unwrap_or_default();
        let score_pct = (r.score * 100.0).round() as u64;
        out.push_str(&format!("{}. {}{}: {}-{}{} (score: {}%)\n",
            i + 1, r.chunk.file_path, total,
            r.chunk.line_start, r.chunk.line_end, ctx, score_pct));
        if let Some(vs) = r.vector_score {
            out.push_str(&format!("   BM25: {:.2}  Vector: {:.4}\n", r.bm25_raw, vs));
        } else {
            out.push_str(&format!("   BM25: {:.2}\n", r.bm25_raw));
        }
        for line in r.chunk.content.split('\n').take(30) {
            out.push_str(&format!("   {}\n", line));
        }
        out.push('\n');
    }
    out
}
