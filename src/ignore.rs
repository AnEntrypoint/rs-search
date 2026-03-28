use std::collections::HashSet;
use std::fs;
use std::path::Path;

pub static IGNORED_DIRS: &[&str] = &[
    "node_modules","bower_components","jspm_packages","web_modules",
    ".git",".svn",".hg",".bzr",
    ".claude",".cursor",".aider",
    ".vscode",".idea",".vs",".atom",
    "dist","dist-server","dist-ssr","dist-client",
    "build","built","out","out-tsc","target",
    "storybook-static",".docusaurus",".gatsby",".vuepress",".nuxt",".next",".tsc",
    ".cache",".parcel-cache",".vite",".turbo",
    ".npm",".yarn",".pnp",".pnpm-store",".rush",".lerna",".nx",
    "coverage",".nyc_output",".coverage","htmlcov","test-results",
    "__tests__","__mocks__","__snapshots__","__fixtures__",
    "cypress","playwright",
    ".tox",".eggs",".hypothesis",".pyre",".pytype",
    "__pycache__",".pytest_cache",".mypy_cache",".venv","venv",
    ".gradle",".mvn",
    "Pods","DerivedData",".bundle","xcuserdata",
    "pkg",".terraform",".terragrunt-cache",".pulumi",".serverless",
    ".firebase",".aws",".azure",".gcloud",".vercel",".netlify",
    "temp","tmp",".tmp",".temp",
    ".llamaindex",".chroma",".vectorstore",".embeddings",
    ".langchain",".autogen",".semantic-kernel",".openai-cache",
    ".anthropic-cache","embeddings","vector-db","faiss-index",
    "chromadb","pinecone-cache","weaviate-data",
    ".pnpm",".bun",
    "assets","static","public","wwwroot","www",
    "cmake_build_debug","cmake_build_release","CMakeFiles",
    ".code-search",
];

pub static CODE_EXTENSIONS: &[&str] = &[
    ".js",".jsx",".ts",".tsx",".mjs",".cjs",".mts",".cts",
    ".py",".pyw",".pyi",
    ".java",
    ".c",".cpp",".cc",".cxx",".h",".hpp",".hh",".hxx",
    ".cs",".csx",
    ".go",
    ".rs",
    ".rb",".erb",
    ".php",".phtml",
    ".swift",
    ".kt",".kts",
    ".scala",".sc",
    ".pl",".pm",
    ".sh",".bash",".zsh",".fish",
    ".ps1",".psm1",".psd1",
    ".lua",
    ".r",
    ".jl",
    ".dart",
    ".ex",".exs",
    ".erl",".hrl",
    ".hs",".lhs",
    ".clj",".cljs",".cljc",
    ".lisp",".lsp",".scm",".ss",".rkt",
    ".f",".for",".f90",".f95",".f03",
    ".asm",".s",
    ".groovy",".gvy",
    ".vb",".vbs",
    ".fs",".fsx",
    ".ml",".mli",
    ".mm",
    ".ino",
    ".vue",
    ".svelte",
    ".coffee",
    ".re",".rei",
    ".xml",".xsd",".html",".htm",".yml",".yaml",".toml",
    ".css",".scss",".sass",".less",
    ".sql",
    ".md",".markdown",".txt",
];

static BINARY_EXTENSIONS: &[&str] = &[
    ".zip",".tar",".gz",".rar",".7z",".iso",
    ".exe",".dll",".so",".dylib",".bin",
    ".jpg",".jpeg",".png",".gif",".bmp",".svg",".ico",
    ".mp3",".mp4",".mov",".avi",".flv",".m4a",
    ".pdf",".doc",".docx",".xls",".xlsx",
    ".woff",".woff2",".ttf",".otf",".eot",
];

pub fn is_code_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    let ext = lower.rfind('.').map(|i| &lower[i..]).unwrap_or("");
    if ext.is_empty() { return false; }
    CODE_EXTENSIONS.contains(&ext)
}

pub fn is_binary_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    let ext = lower.rfind('.').map(|i| &lower[i..]).unwrap_or("");
    BINARY_EXTENSIONS.contains(&ext)
}

pub fn should_ignore_dir(name: &str) -> bool {
    IGNORED_DIRS.contains(&name)
}

fn parse_ignore_file(content: &str) -> HashSet<String> {
    let mut patterns = HashSet::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('!') { continue; }
        let line = line.trim_end_matches('/');
        let line = if line.contains('*') { line.trim_end_matches(|c| c == '*' || c == '/') } else { line };
        if !line.is_empty() { patterns.insert(line.to_string()); }
    }
    patterns
}

pub fn load_ignore_patterns(root: &Path) -> HashSet<String> {
    let mut merged = HashSet::new();
    merged.insert("package-lock.json".into());
    merged.insert("yarn.lock".into());
    merged.insert("pnpm-lock.yaml".into());
    merged.insert("Gemfile.lock".into());
    merged.insert("poetry.lock".into());
    merged.insert("Pipfile.lock".into());
    merged.insert("Cargo.lock".into());
    merged.insert("composer.lock".into());
    merged.insert("go.sum".into());
    merged.insert(".DS_Store".into());
    merged.insert("Thumbs.db".into());
    merged.insert("desktop.ini".into());
    merged.insert(".tern-port".into());

    for name in &[".gitignore",".dockerignore",".npmignore",".eslintignore",".prettierignore",".thornsignore",".codesearchignore"] {
        let p = root.join(name);
        if let Ok(content) = fs::read_to_string(&p) {
            merged.extend(parse_ignore_file(&content));
        }
    }
    merged
}

pub fn should_ignore(rel_path: &str, patterns: &HashSet<String>, is_dir: bool) -> bool {
    let norm = rel_path.replace('\\', "/");
    let parts: Vec<&str> = norm.split('/').collect();
    let name = parts.last().copied().unwrap_or("");

    if is_dir {
        if should_ignore_dir(name) { return true; }
        for pat in patterns { if !pat.contains('/') && name == pat { return true; } }
        return false;
    }

    for part in &parts[..parts.len().saturating_sub(1)] {
        if should_ignore_dir(part) { return true; }
    }

    if !is_code_file(&norm) { return true; }
    if is_binary_file(name) { return true; }

    for pat in patterns {
        if pat.contains('/') {
            if norm.contains(pat.as_str()) { return true; }
        } else if name == pat.as_str() {
            return true;
        } else {
            for part in &parts { if *part == pat.as_str() { return true; } }
        }
    }
    false
}
