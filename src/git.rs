use std::path::Path;
use std::process::Command;
use crate::ignore::{CODE_EXTENSIONS, should_ignore_dir};

pub struct CommitInfo {
    pub hash: String,
    pub message: String,
    pub diff_text: String,
}

fn is_indexed_file(rel_path: &str) -> bool {
    let norm = rel_path.replace('\\', "/");
    let parts: Vec<&str> = norm.split('/').collect();
    for part in &parts[..parts.len().saturating_sub(1)] {
        if should_ignore_dir(part) { return false; }
    }
    let lower = norm.to_lowercase();
    let ext = lower.rfind('.').map(|i| &lower[i..]).unwrap_or("");
    if ext.is_empty() { return false; }
    CODE_EXTENSIONS.contains(&ext)
}

pub fn scan_git_commits(root: &Path, limit: usize) -> Vec<CommitInfo> {
    let hashes = {
        let out = Command::new("git")
            .args(["log", "--format=%H", &format!("-{}", limit)])
            .current_dir(root)
            .output();
        match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            _ => return vec![],
        }
    };

    let mut commits = Vec::new();
    for hash in hashes.lines().filter(|h| !h.is_empty()) {
        let msg_out = Command::new("git")
            .args(["log", "--format=%s%n%b", "-1", hash])
            .current_dir(root)
            .output();
        let message = match msg_out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
            _ => continue,
        };

        let diff_out = Command::new("git")
            .args(["diff-tree", "--no-commit-id", "-r", "--unified=3", hash])
            .current_dir(root)
            .output();
        let raw_diff = match diff_out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
            _ => String::new(),
        };

        let diff_text = filter_diff_to_indexed_files(&raw_diff);
        if diff_text.is_empty() && message.is_empty() { continue; }

        commits.push(CommitInfo {
            hash: hash.to_string(),
            message,
            diff_text,
        });
    }
    commits
}

fn filter_diff_to_indexed_files(raw: &str) -> String {
    let mut out = String::new();
    let mut in_indexed = false;
    let mut current_file_block = String::new();

    for line in raw.lines() {
        if line.starts_with("diff --git ") {
            if in_indexed && !current_file_block.is_empty() {
                out.push_str(&current_file_block);
            }
            current_file_block = String::new();
            let parts: Vec<&str> = line.splitn(4, ' ').collect();
            let file_path = if parts.len() >= 4 {
                parts[3].trim_start_matches("b/")
            } else { "" };
            in_indexed = is_indexed_file(file_path);
            if in_indexed { current_file_block.push_str(line); current_file_block.push('\n'); }
        } else if in_indexed {
            current_file_block.push_str(line);
            current_file_block.push('\n');
            if current_file_block.len() > 4096 {
                out.push_str(&current_file_block[..4096]);
                out.push('\n');
                current_file_block.clear();
                in_indexed = false;
            }
        }
    }
    if in_indexed && !current_file_block.is_empty() {
        out.push_str(&current_file_block);
    }
    out.truncate(8192);
    out
}

pub fn commits_to_searchable(commits: &[CommitInfo]) -> Vec<(String, String)> {
    commits.iter().map(|c| {
        let text = if c.diff_text.is_empty() {
            c.message.clone()
        } else {
            format!("{}\n{}", c.message, &c.diff_text[..c.diff_text.len().min(2048)])
        };
        (c.hash.clone(), text)
    }).collect()
}
