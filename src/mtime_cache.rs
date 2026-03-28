use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

pub struct MtimeCache {
    path: PathBuf,
    map: HashMap<String, u64>,
}

impl MtimeCache {
    pub fn load(db_path: &Path) -> Self {
        let path = db_path.join("mtime-index.json");
        let map = fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Self { path, map }
    }

    pub fn get(&self, file_path: &str) -> Option<u64> {
        self.map.get(file_path).copied()
    }

    pub fn remove(&mut self, file_path: &str) {
        self.map.remove(file_path);
    }

    pub fn insert(&mut self, file_path: String, mtime: u64) {
        self.map.insert(file_path, mtime);
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string(&self.map) {
            let _ = fs::write(&self.path, json);
        }
    }
}
