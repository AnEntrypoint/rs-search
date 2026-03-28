use std::fs;
use std::path::Path;
use walkdir::WalkDir;
use crate::ignore::{load_ignore_patterns, should_ignore, should_ignore_dir};

#[derive(Clone)]
pub struct Chunk {
    pub file_path: String,
    pub chunk_index: usize,
    pub content: String,
    pub line_start: usize,
    pub line_end: usize,
    pub mtime: u64,
}

pub fn scan_repository(root: &Path) -> Vec<Chunk> {
    let patterns = load_ignore_patterns(root);
    let mut chunks = Vec::new();

    for entry in WalkDir::new(root).min_depth(1).into_iter() {
        let entry = match entry { Ok(e) => e, Err(_) => continue };
        let full = entry.path();
        let rel = match full.strip_prefix(root) {
            Ok(r) => r.to_string_lossy().replace('\\', "/"),
            Err(_) => continue,
        };

        if entry.file_type().is_dir() {
            if let Some(name) = full.file_name().and_then(|n| n.to_str()) {
                if should_ignore_dir(name) { continue; }
            }
            if should_ignore(&rel, &patterns, true) { continue; }
            continue;
        }

        if !entry.file_type().is_file() { continue; }
        if should_ignore(&rel, &patterns, false) { continue; }

        let meta = match fs::metadata(full) { Ok(m) => m, Err(_) => continue };
        if meta.len() > 5 * 1024 * 1024 { continue; }

        let mtime = meta.modified().ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let content = match fs::read_to_string(full) { Ok(c) => c, Err(_) => continue };
        let lines: Vec<&str> = content.split('\n').collect();

        if lines.len() <= 60 {
            chunks.push(Chunk {
                file_path: rel,
                chunk_index: 0,
                content,
                line_start: 1,
                line_end: lines.len(),
                mtime,
            });
        } else {
            let chunk_size = 60usize;
            let overlap = 15usize;
            let step = chunk_size - overlap;
            let mut i = 0usize;
            let mut idx = 0usize;
            loop {
                let end = (i + chunk_size).min(lines.len());
                let chunk_content = lines[i..end].join("\n");
                if !chunk_content.trim().is_empty() {
                    chunks.push(Chunk {
                        file_path: rel.clone(),
                        chunk_index: idx,
                        content: chunk_content,
                        line_start: i + 1,
                        line_end: end,
                        mtime,
                    });
                    idx += 1;
                }
                if end == lines.len() { break; }
                i += step;
            }
        }
    }

    chunks
}
