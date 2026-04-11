use std::fs;
use std::io::Write;
use std::path::Path;

pub fn model_path(models_dir: &Path) -> std::path::PathBuf {
    models_dir.join("nomic-embed-text-v1.5.Q4_K_M.gguf")
}

pub fn ensure_assembled(models_dir: &Path) -> Result<std::path::PathBuf, String> {
    let out = model_path(models_dir);
    if out.exists() {
        return Ok(out);
    }
    let mut parts: Vec<std::path::PathBuf> = fs::read_dir(models_dir)
        .map_err(|e| format!("read models dir: {}", e))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("nomic-embed-text-v1.5.Q4_K_M.gguf.part"))
                .unwrap_or(false)
        })
        .collect();
    if parts.is_empty() {
        return Err(format!("no .gguf.part files found in {}", models_dir.display()));
    }
    parts.sort_by_key(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_prefix("nomic-embed-text-v1.5.Q4_K_M.gguf.part"))
            .and_then(|n| n.parse::<u32>().ok())
            .unwrap_or(0)
    });
    let mut out_file = fs::File::create(&out)
        .map_err(|e| format!("create output: {}", e))?;
    for part in &parts {
        let data = fs::read(part).map_err(|e| format!("read {}: {}", part.display(), e))?;
        out_file.write_all(&data).map_err(|e| format!("write: {}", e))?;
    }
    Ok(out)
}
