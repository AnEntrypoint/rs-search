use regex::Regex;
use std::sync::LazyLock;

static SKIP: &[&str] = &["if","for","while","switch","catch","else","return","await","do","match","yield","typeof","new","delete","void","in","of"];
static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:^|\s)(?:async\s+)?(?:function\s+(\w+)|class\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(|(?:static\s+)?(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{|fn\s+(\w+)|struct\s+(\w+)|impl\s+(\w+))").expect("static regex"));

fn looks_like_continued_signature(line: &str) -> bool {
    let trimmed = line.trim_end();
    trimmed.ends_with('(') || trimmed.ends_with(',')
}

const MAX_SIGNATURE_JOIN: usize = 8;

pub fn find_enclosing_context(content: &str, line_start: usize) -> Option<String> {
    if content.trim().is_empty() { return None; }
    let lines: Vec<&str> = content.split('\n').collect();
    if lines.is_empty() { return None; }
    let target = (line_start.saturating_sub(1)).min(lines.len() - 1);
    let mut i = target as isize;
    while i >= 0 {
        let idx = i as usize;
        let mut k = idx;
        let mut joined: Option<String> = None;
        while looks_like_continued_signature(lines[k]) && k > 0 && idx - k < MAX_SIGNATURE_JOIN {
            k -= 1;
            let tail = joined.as_deref().unwrap_or(lines[idx]);
            joined = Some(format!("{} {}", lines[k].trim_end(), tail.trim_start()));
        }
        let candidate = joined.as_deref().unwrap_or(lines[idx]);
        if let Some(caps) = RE.captures(candidate) {
            for j in 1..caps.len() {
                if let Some(m) = caps.get(j) {
                    let name = m.as_str();
                    if !SKIP.contains(&name) { return Some(name.to_string()); }
                }
            }
        }
        i = k as isize - 1;
    }
    None
}

#[cfg(test)]
mod context_eval {
    use super::find_enclosing_context;

    #[test]
    fn detects_real_scopes() {
        let js = "function outerFn() {\n  let x = 1;\n}";
        assert_eq!(find_enclosing_context(js, 2).as_deref(), Some("outerFn"));
        let rs = "impl Searcher {\n    fn search(&self) {\n        let y = 2;\n    }\n}";
        let scope = find_enclosing_context(rs, 3);
        assert!(scope.as_deref() == Some("search") || scope.as_deref() == Some("Searcher"),
            "rust method scope expected, got {:?}", scope);
        let cls = "class Widget {\n  render() {\n    let z = 3;\n  }\n}";
        let scope = find_enclosing_context(cls, 3);
        assert!(scope.as_deref() == Some("render") || scope.as_deref() == Some("Widget"),
            "class method body scope expected, got {:?}", scope);
    }

    #[test]
    fn does_not_mislabel_control_flow_as_scope() {
        let js = "function host() {\n  if (cond) {\n    work();\n  }\n}";
        assert_eq!(find_enclosing_context(js, 3).as_deref(), Some("host"),
            "the enclosing scope of a statement inside an if-block is the function, not the control keyword");
    }
}
