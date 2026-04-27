use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::collections::{HashMap, HashSet};

pub struct EmbedCache {
    dir: PathBuf,
    mem: Mutex<HashMap<String, Vec<f32>>>,
}

impl EmbedCache {
    pub fn new(db_path: &Path) -> Self {
        let dir = db_path.join("emb-cache");
        let _ = fs::create_dir_all(&dir);
        Self { dir, mem: Mutex::new(HashMap::new()) }
    }

    pub fn key(model_tag: &str, dim: usize, text: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(model_tag.as_bytes());
        hasher.update(&dim.to_le_bytes());
        hasher.update(text.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    fn disk_path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{}.bin", key))
    }

    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        if let Ok(mem) = self.mem.lock() {
            if let Some(v) = mem.get(key) { return Some(v.clone()); }
        }
        let path = self.disk_path(key);
        let bytes = fs::read(&path).ok()?;
        if bytes.len() % 4 != 0 { return None; }
        let mut out = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        if let Ok(mut mem) = self.mem.lock() { mem.insert(key.to_string(), out.clone()); }
        Some(out)
    }

    pub fn put(&self, key: &str, vec: &[f32]) {
        if let Ok(mut mem) = self.mem.lock() { mem.insert(key.to_string(), vec.to_vec()); }
        let mut bytes = Vec::with_capacity(vec.len() * 4);
        for v in vec { bytes.extend_from_slice(&v.to_le_bytes()); }
        let _ = fs::write(self.disk_path(key), &bytes);
    }

    pub fn sweep_orphans(&self, live_keys: &HashSet<String>) -> (usize, u64) {
        let mut removed = 0usize;
        let mut bytes_freed = 0u64;
        let entries = match fs::read_dir(&self.dir) { Ok(e) => e, Err(_) => return (0, 0) };
        for ent in entries.flatten() {
            let path = ent.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            let Some(key) = name.strip_suffix(".bin") else { continue };
            if !live_keys.contains(key) {
                let size = ent.metadata().map(|m| m.len()).unwrap_or(0);
                if fs::remove_file(&path).is_ok() {
                    removed += 1;
                    bytes_freed += size;
                }
            }
        }
        if let Ok(mut mem) = self.mem.lock() {
            mem.retain(|k, _| live_keys.contains(k));
        }
        (removed, bytes_freed)
    }
}
