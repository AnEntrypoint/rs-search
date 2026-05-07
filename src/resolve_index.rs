use std::path::{Path, PathBuf};

pub fn resolve_root(root: Option<&str>) -> PathBuf {
    let p = match root {
        Some(s) if !s.is_empty() => PathBuf::from(s),
        _ => std::env::current_dir().expect("cwd"),
    };
    if p.is_absolute() { p } else { std::fs::canonicalize(&p).unwrap_or(p) }
}

pub fn resolve_index(root: &Path, index: Option<&str>, discipline: Option<&str>) -> PathBuf {
    if let Some(idx) = index.filter(|s| !s.is_empty()) {
        let p = PathBuf::from(idx);
        return if p.is_absolute() { p } else { root.join(p) };
    }
    if let Some(name) = discipline.filter(|s| !s.is_empty()) {
        return root.join(".gm").join("disciplines").join(name).join("code-search");
    }
    root.join(".gm").join("code-search")
}
