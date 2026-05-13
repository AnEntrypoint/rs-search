#![cfg(target_arch = "wasm32")]

use serde::Deserialize;

extern "C" {
    pub fn host_vec_search(query_ptr: *const u8, query_len: u32, k: u32) -> u64;
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
