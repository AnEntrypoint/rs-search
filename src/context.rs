use regex::Regex;
use std::sync::LazyLock;

static SKIP: &[&str] = &["if","for","while","switch","catch","else","return","await","do","match","yield","typeof","new","delete","void","in","of"];
static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:^|\s)(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(|(?:static\s+)?(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{|fn\s+(\w+)|struct\s+(\w+)|impl\s+(\w+))").expect("static regex"));

pub fn find_enclosing_context(content: &str, line_start: usize) -> Option<String> {
    let lines: Vec<&str> = content.split('\n').collect();
    let target = (line_start.saturating_sub(1)).min(lines.len().saturating_sub(1));
    for i in (0..=target).rev() {
        if let Some(caps) = RE.captures(lines[i]) {
            for j in 1..caps.len() {
                if let Some(m) = caps.get(j) {
                    let name = m.as_str();
                    if !SKIP.contains(&name) { return Some(name.to_string()); }
                }
            }
        }
    }
    None
}

pub fn get_file_total_lines(root: &std::path::Path, rel_path: &str) -> Option<usize> {
    let full = root.join(rel_path.replace('/', std::path::MAIN_SEPARATOR_STR));
    std::fs::read_to_string(full).ok().map(|c| c.split('\n').count())
}
