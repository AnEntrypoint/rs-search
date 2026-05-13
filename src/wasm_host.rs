#![cfg(target_arch = "wasm32")]

use serde::Deserialize;

extern "C" {
    pub fn host_vec_search(query_ptr: u32, query_len: u32, k: u32) -> u64;
    pub fn host_kv_get(key_ptr: u32, key_len: u32) -> u64;
    pub fn host_kv_put(key_ptr: u32, key_len: u32, val_ptr: u32, val_len: u32) -> u32;
    pub fn host_log(ptr: u32, len: u32);
    pub fn host_now_ms() -> u64;
}

#[inline]
pub fn pack(ptr: u32, len: u32) -> u64 {
    ((len as u64) << 32) | (ptr as u64)
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
    Vec::from_raw_parts(ptr as *mut u8, len as usize, len as usize)
}

pub fn log(msg: &str) {
    let bytes = msg.as_bytes();
    unsafe { host_log(bytes.as_ptr() as u32, bytes.len() as u32) }
}

pub fn now_ms() -> u64 {
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
    let bytes = query.as_bytes();
    let packed = unsafe { host_vec_search(bytes.as_ptr() as u32, bytes.len() as u32, k) };
    let raw = unsafe { take_bytes(packed) };
    if raw.is_empty() {
        return Ok(Vec::new());
    }
    serde_json::from_slice::<Vec<HostHit>>(&raw)
        .map_err(|e| format!("host_vec_search decode: {}", e))
}

pub fn kv_get(key: &str) -> Option<Vec<u8>> {
    let kb = key.as_bytes();
    let packed = unsafe { host_kv_get(kb.as_ptr() as u32, kb.len() as u32) };
    let raw = unsafe { take_bytes(packed) };
    if raw.is_empty() { None } else { Some(raw) }
}

pub fn kv_put(key: &str, val: &[u8]) -> bool {
    let kb = key.as_bytes();
    let r = unsafe {
        host_kv_put(
            kb.as_ptr() as u32,
            kb.len() as u32,
            val.as_ptr() as u32,
            val.len() as u32,
        )
    };
    r != 0
}
