#![cfg(target_arch = "wasm32")]

use serde::Deserialize;

extern "C" {
    pub fn host_vec_search(query_ptr: *const u8, query_len: u32, k: u32) -> u64;
    pub fn host_bm25_search(query_ptr: *const u8, query_len: u32, k: u32, root_ptr: *const u8, root_len: u32) -> u64;
    pub fn host_git_search(query_ptr: *const u8, query_len: u32, k: u32, root_ptr: *const u8, root_len: u32) -> u64;
    pub fn host_log(level: u32, msg_ptr: *const u8, msg_len: u32) -> u32;
    pub fn host_now_ms() -> i64;
}

#[inline]
pub fn unpack(packed: u64) -> (u32, u32) {
    let ptr = (packed & 0xFFFF_FFFF) as u32;
    let len = (packed >> 32) as u32;
    (ptr, len)
}

unsafe fn take_bytes(packed: u64) -> Vec<u8> {
    let (ptr, len) = unpack(packed);
    if ptr == 0 || len == 0 {
        return Vec::new();
    }
    let slice = core::slice::from_raw_parts(ptr as *const u8, len as usize);
    slice.to_vec()
}

unsafe fn take_string(packed: u64) -> Option<String> {
    let bytes = take_bytes(packed);
    if bytes.is_empty() { None } else { Some(String::from_utf8_lossy(&bytes).into_owned()) }
}

pub fn log(msg: &str) {
    let _ = unsafe { host_log(1, msg.as_ptr(), msg.len() as u32) };
}

pub fn now_ms() -> i64 {
    unsafe { host_now_ms() }
}

#[derive(Deserialize, Debug, Clone)]
pub struct HostHit {
    pub id: String,
    pub score: f32,
    #[serde(default)]
    pub payload: serde_json::Value,
}

pub fn vec_search(query: &str, k: u32) -> Result<Vec<HostHit>, String> {
    let packed = unsafe { host_vec_search(query.as_ptr(), query.len() as u32, k) };
    let raw = unsafe { take_bytes(packed) };
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_slice::<Vec<HostHit>>(&raw)
        .map_err(|e| format!("host_vec_search decode: {}", e))
}

pub fn bm25_search(query: &str, k: u32, root: &str) -> Result<Vec<HostHit>, String> {
    let packed = unsafe { host_bm25_search(query.as_ptr(), query.len() as u32, k, root.as_ptr(), root.len() as u32) };
    let raw = unsafe { take_bytes(packed) };
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_slice::<Vec<HostHit>>(&raw)
        .map_err(|e| format!("host_bm25_search decode: {}", e))
}

pub fn git_search(query: &str, k: u32, root: &str) -> Result<Vec<HostHit>, String> {
    let packed = unsafe { host_git_search(query.as_ptr(), query.len() as u32, k, root.as_ptr(), root.len() as u32) };
    let raw = unsafe { take_bytes(packed) };
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_slice::<Vec<HostHit>>(&raw)
        .map_err(|e| format!("host_git_search decode: {}", e))
}

/// Fusion search: combines vector, BM25, and git results with RRF scoring.
/// Mirrors CLI fusion.rs logic — thebird host implements each source.
pub fn fusion_search(query: &str, k: u32, root: &str) -> Vec<HostHit> {
    let vec_hits = vec_search(query, k).unwrap_or_default();
    let bm25_hits = bm25_search(query, k, root).unwrap_or_default();
    let git_hits = git_search(query, k, root).unwrap_or_default();

    // RRF (Reciprocal Rank Fusion) scoring
    let mut scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    let mut payloads: std::collections::HashMap<String, serde_json::Value> = std::collections::HashMap::new();

    for (rank, hit) in vec_hits.iter().enumerate() {
        let rrf = 1.0 / (60.0 + rank as f32);
        *scores.entry(hit.id.clone()).or_insert(0.0) += rrf;
        payloads.entry(hit.id.clone()).or_insert(hit.payload.clone());
    }
    for (rank, hit) in bm25_hits.iter().enumerate() {
        let rrf = 1.0 / (60.0 + rank as f32);
        *scores.entry(hit.id.clone()).or_insert(0.0) += rrf;
        payloads.entry(hit.id.clone()).or_insert(hit.payload.clone());
    }
    for (rank, hit) in git_hits.iter().enumerate() {
        let rrf = 1.0 / (60.0 + rank as f32);
        *scores.entry(hit.id.clone()).or_insert(0.0) += rrf;
        payloads.entry(hit.id.clone()).or_insert(hit.payload.clone());
    }

    let mut results: Vec<HostHit> = scores.into_iter()
        .map(|(id, score)| HostHit {
            payload: payloads.remove(&id).unwrap_or(serde_json::Value::Null),
            id,
            score,
        })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k as usize);
    results
}
