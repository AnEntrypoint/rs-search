use serde::Deserialize;

#[link(wasm_import_module = "env")]
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

const MAX_HOST_RESPONSE_BYTES: u32 = 32 * 1024 * 1024;

// SAFETY: ptr/len cross the wasm-host ABI boundary from plugkit-wasm-wrapper.js,
// a host we control (not arbitrary/untrusted wasm callers). There is no way for
// guest code to independently verify the pointer targets a live, correctly
// aligned allocation of at least `len` bytes -- that guarantee comes entirely
// from the wrapper's own packing (`(ptr | len << 32)` over its own buffer) and
// from the wasm32 linear-memory model, where every host-visible ptr is a
// byte-aligned offset into the single sandboxed memory (so u8 alignment is
// always satisfied). The len sanity bound below is not a validity check --
// it is a defense against a corrupted/misbehaving host packing an absurd len
// and this code reading far past the actual response, capped at a size no
// legitimate search-index blob (hit list, bm25/git result JSON) should exceed.
//
// No free call after the copy: this is the same host_vec_search-family
// convention rs-plugkit's wasm_dispatch.rs uses (unpack_to_value/unpack_to_string
// never free either) -- the JS host retains and manages this buffer itself,
// unlike rs-exec's separate rs_exec_alloc/rs_exec_free pair, which exists only
// for buffers the wasm side itself allocated via rs_exec_alloc.
unsafe fn take_bytes(packed: u64) -> Result<Vec<u8>, String> {
    let (ptr, len) = unpack(packed);
    if ptr == 0 || len == 0 {
        return Ok(Vec::new());
    }
    if len > MAX_HOST_RESPONSE_BYTES {
        return Err(format!("host response len {} exceeds sanity bound {}", len, MAX_HOST_RESPONSE_BYTES));
    }
    let slice = core::slice::from_raw_parts(ptr as *const u8, len as usize);
    Ok(slice.to_vec())
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
    let raw = unsafe { take_bytes(packed) }?;
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_slice::<Vec<HostHit>>(&raw)
        .map_err(|e| format!("host_vec_search decode: {}", e))
}

pub fn bm25_search(query: &str, k: u32, root: &str) -> Result<Vec<HostHit>, String> {
    let packed = unsafe { host_bm25_search(query.as_ptr(), query.len() as u32, k, root.as_ptr(), root.len() as u32) };
    let raw = unsafe { take_bytes(packed) }?;
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_slice::<Vec<HostHit>>(&raw)
        .map_err(|e| format!("host_bm25_search decode: {}", e))
}

pub fn git_search(query: &str, k: u32, root: &str) -> Result<Vec<HostHit>, String> {
    let packed = unsafe { host_git_search(query.as_ptr(), query.len() as u32, k, root.as_ptr(), root.len() as u32) };
    let raw = unsafe { take_bytes(packed) }?;
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_slice::<Vec<HostHit>>(&raw)
        .map_err(|e| format!("host_git_search decode: {}", e))
}

const CANDIDATE_MULTIPLIER: u32 = 5;
const CANDIDATE_FLOOR: u32 = 50;

pub fn fusion_search(query: &str, k: u32, root: &str) -> Vec<HostHit> {
    let cand_k = k.saturating_mul(CANDIDATE_MULTIPLIER).max(CANDIDATE_FLOOR);
    let vec_result = vec_search(query, cand_k);
    let bm25_result = bm25_search(query, cand_k, root);
    let git_result = git_search(query, cand_k, root);

    if let Err(e) = &vec_result {
        log(&format!("search error vec: {}", e));
    }
    if let Err(e) = &bm25_result {
        log(&format!("search error bm25: {}", e));
    }
    if let Err(e) = &git_result {
        log(&format!("search error git: {}", e));
    }
    if let (Err(vec_e), Err(bm25_e), Err(git_e)) = (&vec_result, &bm25_result, &git_result) {
        log(&format!(
            "fusion_search: all backends failed, returning empty results (vec: {}, bm25: {}, git: {})",
            vec_e, bm25_e, git_e
        ));
    }

    let vec_hits = vec_result.unwrap_or_default();
    let bm25_hits = bm25_result.unwrap_or_default();
    let git_hits = git_result.unwrap_or_default();

    let all_hits = [&vec_hits, &bm25_hits, &git_hits];
    let mut payloads: std::collections::HashMap<String, serde_json::Value> = std::collections::HashMap::new();
    for source in &all_hits {
        for hit in source.iter() {
            payloads.entry(hit.id.clone()).or_insert_with(|| hit.payload.clone());
        }
    }

    let ranked_lists: Vec<Vec<String>> = all_hits.iter()
        .map(|source| source.iter().map(|h| h.id.clone()).collect())
        .collect();

    let weights = [1.0, crate::fusion::IDENTIFIER_BOOST, 1.0];
    let mut results: Vec<HostHit> = crate::fusion::fuse_n(&ranked_lists, &weights, query)
        .into_iter()
        .take(k as usize)
        .map(|(id, score)| HostHit {
            payload: payloads.remove(&id).unwrap_or(serde_json::Value::Null),
            id,
            score: score as f32,
        })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.id.cmp(&b.id)));
    results
}
