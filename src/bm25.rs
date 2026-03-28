use std::collections::{HashMap, HashSet};
use regex::Regex;
use crate::scanner::Chunk;

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
}

pub fn search(query: &str, chunks: &[Chunk]) -> Vec<SearchResult> {
    if query.trim().is_empty() { return vec![]; }

    let mut index: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut meta_list: Vec<ChunkMeta> = Vec::with_capacity(chunks.len());

    for (idx, chunk) in chunks.iter().enumerate() {
        let frequency = tokenize_to_frequency(&chunk.content, &mut index, idx);
        let file_name_tokens: HashSet<String> = tokenize(&chunk.file_path).into_iter().collect();
        let symbols = extract_symbols(&chunk.content);
        let is_code = is_code_file(&chunk.file_path);
        meta_list.push(ChunkMeta {
            file_name_tokens,
            symbols,
            frequency,
            is_code,
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
    for token in &query_tokens {
        if let Some(set) = index.get(token) { candidates.extend(set); }
    }
    for sym in &query_symbols {
        if let Some(set) = index.get(sym) { candidates.extend(set); }
    }

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

        if query_tokens.len() > 1 && meta.content_lower.contains(&query_lower) {
            score += 30.0;
        }
        for sym in &query_symbols {
            if meta.symbols.contains(sym) { score += 10.0; }
        }
        let file_matches: usize = query_tokens.iter().filter(|t| meta.file_name_tokens.contains(*t)).count();
        score += file_matches as f64 * 10.0;

        for token in &query_tokens {
            if let Some(doc_set) = index.get(token) {
                if doc_set.contains(&idx) {
                    let tf = (*meta.frequency.get(token).unwrap_or(&1)).min(5) as f64;
                    let token_idf = *idf.get(token).unwrap_or(&1.0);
                    let length_boost = if token.len() > 4 { 1.5 } else { 1.0 };
                    score += length_boost * tf * token_idf;
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
    }).collect();

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(10);
    results
}

fn tokenize_to_frequency(text: &str, index: &mut HashMap<String, HashSet<usize>>, chunk_idx: usize) -> HashMap<String, u32> {
    let mut freq: HashMap<String, u32> = HashMap::new();
    let camel_re = Regex::new(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[0-9]+").unwrap();

    for word in text.split_whitespace() {
        if word != word.to_lowercase() {
            for t in camel_re.find_iter(word) {
                let tok = t.as_str().to_lowercase();
                if tok.len() > 1 { *freq.entry(tok).or_insert(0) += 1; }
            }
        }
        let cleaned: String = word.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect::<String>().to_lowercase();
        if cleaned.len() > 1 {
            *freq.entry(cleaned.clone()).or_insert(0) += 1;
            if word.contains('-') || word.contains('_') || word.contains('.') {
                for part in word.split(|c| c == '-' || c == '_' || c == '.') {
                    let pc: String = part.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
                    if pc.len() > 1 && pc != cleaned { *freq.entry(pc).or_insert(0) += 1; }
                }
            }
        }
    }

    for token in freq.keys() {
        index.entry(token.clone()).or_insert_with(HashSet::new).insert(chunk_idx);
    }
    freq
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: HashSet<String> = HashSet::new();
    let camel_re = Regex::new(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|[0-9]+").unwrap();
    for word in text.split_whitespace() {
        for t in camel_re.find_iter(word) {
            let tok = t.as_str().to_lowercase();
            if tok.len() > 1 { tokens.insert(tok); }
        }
        for part in word.split(|c| c == '-' || c == '_' || c == '.') {
            let pc: String = part.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
            if pc.len() > 1 { tokens.insert(pc); }
        }
        let cleaned: String = word.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect::<String>().to_lowercase();
        if cleaned.len() > 1 { tokens.insert(cleaned); }
    }
    tokens.into_iter().filter(|t| t.len() > 1).collect()
}

fn extract_symbols(text: &str) -> HashSet<String> {
    let mut symbols = HashSet::new();
    let fn_re = Regex::new(r"(?:async\s+)?function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\(").unwrap();
    let class_re = Regex::new(r"class\s+(\w+)").unwrap();
    let export_re = Regex::new(r"export\s+(?:async\s+)?(?:function|class)\s+(\w+)").unwrap();
    let fn_rs = Regex::new(r"fn\s+(\w+)").unwrap();
    let struct_re = Regex::new(r"struct\s+(\w+)").unwrap();
    let impl_re = Regex::new(r"impl\s+(\w+)").unwrap();

    for re in &[&fn_re, &class_re, &export_re, &fn_rs, &struct_re, &impl_re] {
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
