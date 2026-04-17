use std::fs;
use std::path::Path;
use crate::ignore::{load_ignore_patterns, should_ignore, should_ignore_dir};
use crate::pdf::{is_pdf, pdf_chunks};

#[derive(Clone)]
pub struct Chunk {
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub line_start: usize,
    pub line_end: usize,
    pub mtime: u64,
}

fn chunk_text(rel: &str, content: &str, mtime: u64, chunks: &mut Vec<Chunk>) {
    let line_count = content.split('\n').count();
    if line_count <= 60 {
        chunks.push(Chunk { file_path: rel.to_string(), chunk_index: 0, line_end: line_count, content: content.to_string(), line_start: 1, mtime });
        return;
    }
    let lines: Vec<&str> = content.split('\n').collect();
    let step = 60 - 15;
    let mut i = 0usize;
    let mut idx = 0usize;
    loop {
        let end = (i + 60).min(lines.len());
        let chunk_content = lines[i..end].join("\n");
        if !chunk_content.trim().is_empty() {
            chunks.push(Chunk { file_path: rel.to_string(), chunk_index: idx, content: chunk_content, line_start: i + 1, line_end: end, mtime });
            idx += 1;
        }
        if end == lines.len() { break; }
        i += step;
    }
}

fn walk(dir: &Path, root: &Path, patterns: &std::collections::HashSet<String>, db_path: Option<&Path>, chunks: &mut Vec<Chunk>) {
    let entries = match fs::read_dir(dir) { Ok(e) => e, Err(_) => return };
    for entry in entries.flatten() {
        let full = entry.path();
        let rel = match full.strip_prefix(root) {
            Ok(r) => r.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };
        let ft = match entry.file_type() { Ok(t) => t, Err(_) => continue };
        if ft.is_dir() {
            let name = full.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if should_ignore_dir(name) || should_ignore(&rel, patterns, true) { continue; }
            walk(&full, root, patterns, db_path, chunks);
            continue;
        }
        if !ft.is_file() { continue; }
        if should_ignore(&rel, patterns, false) { continue; }
        let meta = match fs::metadata(&full) { Ok(m) => m, Err(_) => continue };
        if meta.len() > 50 * 1024 * 1024 { continue; }
        let mtime = meta.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs()).unwrap_or(0);
        if is_pdf(&rel) {
            chunks.extend(pdf_chunks(&rel, &full, mtime, db_path));
            continue;
        }
        let content = match fs::read_to_string(&full) { Ok(c) => c, Err(_) => continue };
        chunk_text(&rel, &content, mtime, chunks);
    }
}

pub fn scan_repository(root: &Path) -> Vec<Chunk> {
    let patterns = load_ignore_patterns(root);
    let db_path = root.join(".code-search");
    let db = if db_path.exists() { Some(db_path.as_path()) } else { None };
    let mut chunks = Vec::new();
    walk(root, root, &patterns, db, &mut chunks);
    chunks
}
