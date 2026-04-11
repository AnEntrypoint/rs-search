use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use regex::Regex;
use crate::scanner::Chunk;

static SYMBOL_RES: OnceLock<Vec<Regex>> = OnceLock::new();

fn symbol_res() -> &'static Vec<Regex> {
    SYMBOL_RES.get_or_init(|| vec![
        Regex::new(r"(?:async\s+)?function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\(").unwrap(),
        Regex::new(r"class\s+(\w+)").unwrap(),
        Regex::new(r"export\s+(?:async\s+)?(?:function|class)\s+(\w+)").unwrap(),
        Regex::new(r"fn\s+(\w+)").unwrap(),
        Regex::new(r"struct\s+(\w+)").unwrap(),
        Regex::new(r"impl\s+(\w+)").unwrap(),
    ])
}

struct ChunkMeta {
    file_name_tokens: HashSet<String>,
    symbols: HashSet<String>,
    frequency: HashMap<String, u32>,
    is_code: bool,
    content_lower: String,
}

pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f64,
    pub bm25_raw: f64,
    pub vector_score: Option<f32>,
}

fn split_camel(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for i in 0..chars.len() {
        let c = chars[i];
        let is_upper = c.is_uppercase();
        let prev_lower = i > 0 && chars[i-1].is_lowercase();
        let next_lower = i + 1 < chars.len() && chars[i+1].is_lowercase();
        if is_upper && (prev_lower || next_lower) && !cur.is_empty() {
            if cur.len() > 1 { tokens.push(cur.to_lowercase()); }
            cur = c.to_string();
        } else {
            cur.push(c);
        }
    }
    if cur.len() > 1 { tokens.push(cur.to_lowercase()); }
    tokens
}

fn add_word_tokens(word: &str, out: &mut HashSet<String>) {
    if word != word.to_lowercase() {
        for t in split_camel(word) { if t.len() > 1 { out.insert(t); } }
    }
    for part in word.split(|c: char| c == '-' || c == '_' || c == '.') {
        let pc: String = part.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        if pc.len() > 1 { out.insert(pc); }
    }
    let cleaned: String = word.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect::<String>().to_lowercase();
    if cleaned.len() > 1 { out.insert(cleaned); }
}

pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: HashSet<String> = HashSet::new();
    for word in text.split_whitespace() { add_word_tokens(word, &mut tokens); }
    tokens.into_iter().collect()
}

fn tokenize_to_frequency(text: &str, index: &mut HashMap<String, HashSet<usize>>, chunk_idx: usize) -> HashMap<String, u32> {
    let mut freq: HashMap<String, u32> = HashMap::new();
    for word in text.split_whitespace() {
        let mut word_tokens: HashSet<String> = HashSet::new();
        add_word_tokens(word, &mut word_tokens);
        for tok in word_tokens { *freq.entry(tok).or_insert(0) += 1; }
    }
    for token in freq.keys() {
        index.entry(token.clone()).or_default().insert(chunk_idx);
    }
    freq
}

fn extract_symbols(text: &str) -> HashSet<String> {
    let mut symbols = HashSet::new();
    for re in symbol_res() {
        for cap in re.captures_iter(text) {
            for i in 1..cap.len() {
                if let Some(m) = cap.get(i) { symbols.insert(m.as_str().to_lowercase()); }
            }
        }
    }
    symbols
}

fn is_code_file(path: &str) -> bool {
    let code_exts = [".js",".ts",".jsx",".tsx",".py",".java",".go",".rs",".rb",".cs",".cpp",".c",".swift",".kt",".php"];
    let lower = path.to_lowercase();
    let ext = lower.rfind('.').map(|i| &lower[i..]).unwrap_or("");
    code_exts.contains(&ext)
}

pub fn search(query: &str, chunks: &[Chunk]) -> Vec<SearchResult> {
    if query.trim().is_empty() { return vec![]; }

    let mut index: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut meta_list: Vec<ChunkMeta> = Vec::with_capacity(chunks.len());

    for (idx, chunk) in chunks.iter().enumerate() {
        let frequency = tokenize_to_frequency(&chunk.content, &mut index, idx);
        let file_name_tokens: HashSet<String> = tokenize(&chunk.file_path).into_iter().collect();
        let symbols = extract_symbols(&chunk.content);
        meta_list.push(ChunkMeta {
            file_name_tokens, symbols, frequency,
            is_code: is_code_file(&chunk.file_path),
            content_lower: chunk.content.to_lowercase(),
        });
    }

    let n = chunks.len();
    let mut idf: HashMap<String, f64> = HashMap::new();
    for (token, doc_set) in &index {
        idf.insert(token.clone(), ((n + 1) as f64 / (doc_set.len() + 1) as f64).ln() + 1.0);
    }

    let query_tokens = tokenize(query);
    let query_symbols = extract_symbols(query);
    let query_lower = query.to_lowercase();

    let mut candidates: HashSet<usize> = HashSet::new();
    for token in &query_tokens { if let Some(set) = index.get(token) { candidates.extend(set); } }
    for sym in &query_symbols { if let Some(set) = index.get(sym) { candidates.extend(set); } }

    let scoring_candidates: Vec<usize> = if candidates.len() > 500 {
        let mut ranked: Vec<usize> = candidates.into_iter().collect();
        ranked.sort_by(|&a, &b| {
            let sum = |i: usize| -> f64 {
                query_tokens.iter().filter_map(|t| {
                    index.get(t).filter(|s| s.contains(&i)).map(|_| *idf.get(t).unwrap_or(&1.0))
                }).sum()
            };
            sum(b).partial_cmp(&sum(a)).unwrap_or(std::cmp::Ordering::Equal)
        });
        ranked.truncate(500);
        ranked
    } else {
        candidates.into_iter().collect()
    };

    let mut scores: Vec<(usize, f64)> = Vec::new();
    for idx in scoring_candidates {
        let meta = &meta_list[idx];
        let mut score = 0.0f64;
        if query_tokens.len() > 1 && meta.content_lower.contains(&query_lower) { score += 30.0; }
        for sym in &query_symbols { if meta.symbols.contains(sym) { score += 10.0; } }
        let file_matches = query_tokens.iter().filter(|t| meta.file_name_tokens.contains(*t)).count();
        score += file_matches as f64 * 10.0;
        for token in &query_tokens {
            if let Some(doc_set) = index.get(token) {
                if doc_set.contains(&idx) {
                    let tf = (*meta.frequency.get(token).unwrap_or(&1)).min(5) as f64;
                    let length_boost = if token.len() > 4 { 1.5 } else { 1.0 };
                    score += length_boost * tf * *idf.get(token).unwrap_or(&1.0);
                }
            }
        }
        if meta.is_code { score *= 1.2; }
        if score > 0.0 { scores.push((idx, score)); }
    }

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_score = scores.first().map(|s| s.1).unwrap_or(1.0);

    let mut best_per_file: HashMap<String, (usize, f64)> = HashMap::new();
    for (idx, raw) in &scores {
        let fp = &chunks[*idx].file_path;
        let entry = best_per_file.entry(fp.clone()).or_insert((*idx, *raw));
        if *raw > entry.1 { *entry = (*idx, *raw); }
    }

    let mut results: Vec<SearchResult> = best_per_file.into_values().map(|(idx, raw)| SearchResult {
        chunk: chunks[idx].clone(),
        score: raw / max_score,
        bm25_raw: raw,
        vector_score: None,
    }).collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(10);
    results
}
