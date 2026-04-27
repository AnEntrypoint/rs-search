use std::fs;
use std::path::Path;
use ignore::WalkBuilder;
use crate::ignore::{is_code_file, is_binary_file, should_ignore_dir};
#[cfg(feature = "pdf")]
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

const CHUNK_LINES: usize = 60;
const CHUNK_OVERLAP: usize = 15;
const DEFAULT_MAX_FILE_BYTES: u64 = 50 * 1024 * 1024;

fn max_file_bytes() -> u64 {
    std::env::var("RS_SEARCH_MAX_FILE_BYTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_FILE_BYTES)
}

fn chunk_text(rel: &str, content: &str, mtime: u64, chunks: &mut Vec<Chunk>) {
    let line_count = content.split('\n').count();
    if line_count <= CHUNK_LINES {
        chunks.push(Chunk {
            file_path: rel.to_string(), chunk_index: 0, line_end: line_count,
            content: content.to_string(), line_start: 1, mtime,
        });
        return;
    }
    let lines: Vec<&str> = content.split('\n').collect();
    let step = CHUNK_LINES - CHUNK_OVERLAP;
    let mut i = 0usize;
    let mut idx = 0usize;
    loop {
        let end = (i + CHUNK_LINES).min(lines.len());
        let body = lines[i..end].join("\n");
        if !body.trim().is_empty() {
            chunks.push(Chunk {
                file_path: rel.to_string(), chunk_index: idx,
                content: body, line_start: i + 1, line_end: end, mtime,
            });
            idx += 1;
        }
        if end == lines.len() { break; }
        i += step;
    }
}

fn path_hits_ignored_dir(rel: &str) -> bool {
    rel.split('/').any(should_ignore_dir)
}

pub fn scan_repository(root: &Path) -> Vec<Chunk> {
    let db_path = root.join(".gm").join("code-search");
    let legacy_db_path = root.join(".code-search");
    let db_ref = if db_path.exists() { Some(db_path.clone()) }
                 else if legacy_db_path.exists() { Some(legacy_db_path) }
                 else { None };
    let db = db_ref.as_deref();
    let mut chunks = Vec::new();
    let walker = WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true)
        .ignore(true)
        .parents(true)
        .add_custom_ignore_filename(".codesearchignore")
        .filter_entry(|e| {
            let name = e.file_name().to_str().unwrap_or("");
            !should_ignore_dir(name)
        })
        .build();
    for result in walker {
        let entry = match result { Ok(e) => e, Err(_) => continue };
        let full = entry.path();
        if full == root { continue; }
        let rel = match full.strip_prefix(root) {
            Ok(r) => r.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };
        let ft = match entry.file_type() { Some(t) => t, None => continue };
        if ft.is_dir() { continue; }
        if !ft.is_file() { continue; }
        if path_hits_ignored_dir(&rel) { continue; }
        let name = full.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if !is_code_file(&rel) { continue; }
        if is_binary_file(name) { continue; }
        let meta = match fs::metadata(full) { Ok(m) => m, Err(_) => continue };
        if meta.len() > max_file_bytes() { continue; }
        let mtime = meta.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs()).unwrap_or(0);
        #[cfg(feature = "pdf")]
        {
            if is_pdf(&rel) {
                chunks.extend(pdf_chunks(&rel, full, mtime, db));
                continue;
            }
        }
        let content = match fs::read_to_string(full) { Ok(c) => c, Err(_) => continue };
        chunk_text(&rel, &content, mtime, &mut chunks);
    }
    chunks
}
