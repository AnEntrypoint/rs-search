use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::collections::{HashMap, HashSet};

use libsql::{Builder, Connection};
use tokio::runtime::{Handle, Runtime};

const SCHEMA: &str = "CREATE TABLE IF NOT EXISTS emb_cache (key TEXT PRIMARY KEY, vec BLOB NOT NULL) WITHOUT ROWID";

pub struct EmbedCache {
    conn: Connection,
    mem: Mutex<HashMap<String, Vec<f32>>>,
}

static FALLBACK_RUNTIME: OnceLock<Runtime> = OnceLock::new();
static SHARED_CONN: OnceLock<Connection> = OnceLock::new();

pub fn set_shared_connection(conn: Connection) {
    let _ = SHARED_CONN.set(conn);
}

pub fn shared_connection() -> Option<Connection> {
    SHARED_CONN.get().cloned()
}

fn fallback_runtime() -> &'static Runtime {
    FALLBACK_RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("rs-search EmbedCache: build current_thread runtime")
    })
}

fn block_on<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    match Handle::try_current() {
        Ok(h) => tokio::task::block_in_place(|| h.block_on(fut)),
        Err(_) => fallback_runtime().block_on(fut),
    }
}

impl EmbedCache {
    pub fn new(db_path: &Path) -> Self {
        if !db_path.exists() {
            let _ = std::fs::create_dir_all(db_path);
        }
        let file = db_path.join("emb-cache.db");
        let conn = block_on(async {
            let db = Builder::new_local(&file)
                .build()
                .await
                .expect("rs-search EmbedCache: open libsql");
            let conn = db.connect().expect("rs-search EmbedCache: connect");
            conn.execute(SCHEMA, ())
                .await
                .expect("rs-search EmbedCache: create schema");
            conn
        });
        Self { conn, mem: Mutex::new(HashMap::new()) }
    }

    pub fn with_connection(conn: Connection) -> Self {
        block_on(async {
            conn.execute(SCHEMA, ())
                .await
                .expect("rs-search EmbedCache: create schema on shared conn");
        });
        Self { conn, mem: Mutex::new(HashMap::new()) }
    }

    pub fn key(model_tag: &str, dim: usize, text: &str) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(model_tag.as_bytes());
        hasher.update(&dim.to_le_bytes());
        hasher.update(text.as_bytes());
        hasher.finalize().to_hex().to_string()
    }

    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        if let Ok(mem) = self.mem.lock() {
            if let Some(v) = mem.get(key) { return Some(v.clone()); }
        }
        let key_owned = key.to_string();
        let conn = self.conn.clone();
        let bytes: Option<Vec<u8>> = block_on(async move {
            let mut rows = conn
                .query("SELECT vec FROM emb_cache WHERE key = ?1", libsql::params![key_owned])
                .await
                .ok()?;
            let row = rows.next().await.ok().flatten()?;
            match row.get_value(0).ok()? {
                libsql::Value::Blob(b) => Some(b),
                _ => None,
            }
        });
        let bytes = bytes?;
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
        let key_owned = key.to_string();
        let conn = self.conn.clone();
        block_on(async move {
            let _ = conn
                .execute(
                    "INSERT OR REPLACE INTO emb_cache (key, vec) VALUES (?1, ?2)",
                    libsql::params![key_owned, bytes],
                )
                .await;
        });
    }

    pub fn sweep_orphans(&self, live_keys: &HashSet<String>) -> (usize, u64) {
        let conn = self.conn.clone();
        let live: Vec<String> = live_keys.iter().cloned().collect();
        let (removed, bytes_freed) = block_on(async move {
            let total_bytes_before: u64 = {
                let mut rows = match conn.query("SELECT COALESCE(SUM(LENGTH(vec)),0) FROM emb_cache", ()).await {
                    Ok(r) => r,
                    Err(_) => return (0usize, 0u64),
                };
                match rows.next().await.ok().flatten() {
                    Some(r) => r.get::<i64>(0).unwrap_or(0) as u64,
                    None => 0,
                }
            };
            let removed_count: usize = if live.is_empty() {
                let mut rows = match conn.query("SELECT COUNT(*) FROM emb_cache", ()).await {
                    Ok(r) => r,
                    Err(_) => return (0, 0),
                };
                let n = rows.next().await.ok().flatten()
                    .and_then(|r| r.get::<i64>(0).ok()).unwrap_or(0) as usize;
                let _ = conn.execute("DELETE FROM emb_cache", ()).await;
                n
            } else {
                let placeholders = (1..=live.len()).map(|i| format!("?{}", i)).collect::<Vec<_>>().join(",");
                let count_sql = format!("SELECT COUNT(*) FROM emb_cache WHERE key NOT IN ({})", placeholders);
                let del_sql = format!("DELETE FROM emb_cache WHERE key NOT IN ({})", placeholders);
                let params: Vec<libsql::Value> = live.iter().map(|k| libsql::Value::Text(k.clone())).collect();
                let mut rows = match conn.query(&count_sql, params.clone()).await {
                    Ok(r) => r,
                    Err(_) => return (0, 0),
                };
                let n = rows.next().await.ok().flatten()
                    .and_then(|r| r.get::<i64>(0).ok()).unwrap_or(0) as usize;
                let _ = conn.execute(&del_sql, params).await;
                n
            };
            let total_bytes_after: u64 = {
                let mut rows = match conn.query("SELECT COALESCE(SUM(LENGTH(vec)),0) FROM emb_cache", ()).await {
                    Ok(r) => r,
                    Err(_) => return (removed_count, 0),
                };
                match rows.next().await.ok().flatten() {
                    Some(r) => r.get::<i64>(0).unwrap_or(0) as u64,
                    None => 0,
                }
            };
            (removed_count, total_bytes_before.saturating_sub(total_bytes_after))
        });
        if let Ok(mut mem) = self.mem.lock() {
            mem.retain(|k, _| live_keys.contains(k));
        }
        (removed, bytes_freed)
    }
}
