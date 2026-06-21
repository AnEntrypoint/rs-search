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
