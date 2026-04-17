use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use serde_json::{json, Value};
use crate::scanner::Chunk;

pub fn is_pdf(rel: &str) -> bool {
    rel.to_ascii_lowercase().ends_with(".pdf")
}

fn cache_key(abs: &Path, mtime: u64) -> String {
    let mut h = DefaultHasher::new();
    abs.to_string_lossy().hash(&mut h);
    mtime.hash(&mut h);
    format!("{:016x}", h.finish())
}

fn cache_dir(db_path: &Path) -> PathBuf {
    db_path.join("pdf-cache")
}

fn read_cache(db_path: &Path, key: &str) -> Option<Vec<String>> {
    let p = cache_dir(db_path).join(format!("{}.json", key));
    let raw = fs::read_to_string(&p).ok()?;
    let v: Value = serde_json::from_str(&raw).ok()?;
    let arr = v.as_array()?;
    Some(arr.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect())
}

fn write_cache(db_path: &Path, key: &str, pages: &[String]) {
    let dir = cache_dir(db_path);
    let _ = fs::create_dir_all(&dir);
    let arr: Vec<Value> = pages.iter().map(|s| json!(s)).collect();
    let _ = fs::write(dir.join(format!("{}.json", key)), Value::Array(arr).to_string());
}

fn extract_pages_raw(abs: &Path) -> Result<Vec<String>, String> {
    let bytes = fs::read(abs).map_err(|e| format!("read pdf: {}", e))?;
    let text = pdf_extract::extract_text_from_mem(&bytes)
        .map_err(|e| format!("pdf-extract: {}", e))?;
    Ok(text.split('\u{000C}').map(|s| s.to_string()).collect())
}

pub fn extract_pages(abs: &Path, db_path: Option<&Path>, mtime: u64) -> Vec<String> {
    if let Some(db) = db_path {
        let key = cache_key(abs, mtime);
        if let Some(cached) = read_cache(db, &key) {
            return cached;
        }
        let pages = match extract_pages_raw(abs) { Ok(p) => p, Err(_) => return Vec::new() };
        write_cache(db, &key, &pages);
        return pages;
    }
    extract_pages_raw(abs).unwrap_or_default()
}

pub fn pdf_chunks(rel: &str, abs: &Path, mtime: u64, db_path: Option<&Path>) -> Vec<Chunk> {
    let pages = extract_pages(abs, db_path, mtime);
    let mut out = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        let trimmed = page.trim();
        if trimmed.is_empty() { continue; }
        let page_num = i + 1;
        out.push(Chunk {
            file_path: rel.to_string(),
            chunk_index: i,
            content: trimmed.to_string(),
            line_start: page_num,
            line_end: page_num,
            mtime,
        });
    }
    out
}
