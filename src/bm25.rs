use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use regex::Regex;
use crate::scanner::Chunk;
use crate::tokenize::{add_word_tokens, tokenize as tok};

pub use crate::tokenize::tokenize;

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

#[derive(Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub score: f64,
    pub bm25_raw: f64,
    pub vector_score: Option<f32>,
}

fn tokenize_to_frequency(text: &str, index: &mut HashMap<String, HashSet<usize>>, chunk_idx: usize) -> HashMap<String, u32> {
    let mut freq: HashMap<String, u32> = HashMap::new();
    for word in text.split_whitespace() {
        let mut word_tokens: HashSet<String> = HashSet::new();
        add_word_tokens(word, &mut word_tokens);
        for t in word_tokens { *freq.entry(t).or_insert(0) += 1; }
    }
    for token in freq.keys() { index.entry(token.clone()).or_default().insert(chunk_idx); }
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

pub fn search_texts(query: &str, items: &[(String, String)]) -> Vec<(String, f64)> {
    if query.trim().is_empty() { return vec![]; }
    let mut index: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut freqs: Vec<HashMap<String, u32>> = Vec::with_capacity(items.len());
    for (idx, (_, text)) in items.iter().enumerate() {
        freqs.push(tokenize_to_frequency(text, &mut index, idx));
    }
    let n = items.len();
    let mut idf: HashMap<String, f64> = HashMap::new();
    for (token, doc_set) in &index {
        idf.insert(token.clone(), ((n + 1) as f64 / (doc_set.len() + 1) as f64).ln() + 1.0);
    }
    let query_tokens = tok(query);
    let mut candidates: HashSet<usize> = HashSet::new();
    for t in &query_tokens { if let Some(s) = index.get(t) { candidates.extend(s); } }
    let mut scores: Vec<(usize, f64)> = candidates.into_iter().filter_map(|idx| {
        let freq = &freqs[idx];
        let text_lower = items[idx].1.to_lowercase();
        let query_lower = query.to_lowercase();
        let mut score = 0.0f64;
        if query_tokens.len() > 1 && text_lower.contains(&query_lower) { score += 30.0; }
        for t in &query_tokens {
            if let Some(ds) = index.get(t) {
                if ds.contains(&idx) {
                    let tf = (*freq.get(t).unwrap_or(&1)).min(5) as f64;
                    let lb = if t.len() > 4 { 1.5 } else { 1.0 };
                    score += lb * tf * *idf.get(t).unwrap_or(&1.0);
                }
            }
        }
        if score > 0.0 { Some((idx, score)) } else { None }
    }).collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max = scores.first().map(|s| s.1).unwrap_or(1.0);
    scores.iter().map(|(idx, raw)| (items[*idx].0.clone(), raw / max)).collect()
}

fn build_index(chunks: &[Chunk]) -> (HashMap<String, HashSet<usize>>, Vec<ChunkMeta>) {
    let mut index: HashMap<String, HashSet<usize>> = HashMap::new();
    let mut meta: Vec<ChunkMeta> = Vec::with_capacity(chunks.len());
    for (idx, c) in chunks.iter().enumerate() {
        let frequency = tokenize_to_frequency(&c.content, &mut index, idx);
        let file_name_tokens: HashSet<String> = tok(&c.file_path).into_iter().collect();
        let symbols = extract_symbols(&c.content);
        meta.push(ChunkMeta {
            file_name_tokens, symbols, frequency,
            is_code: crate::ignore::is_code_file(&c.file_path),
            content_lower: c.content.to_lowercase(),
        });
    }
    (index, meta)
}

fn compute_idf(index: &HashMap<String, HashSet<usize>>, n: usize) -> HashMap<String, f64> {
    let mut idf: HashMap<String, f64> = HashMap::new();
    for (t, ds) in index {
        idf.insert(t.clone(), ((n + 1) as f64 / (ds.len() + 1) as f64).ln() + 1.0);
    }
    idf
}

fn prune_candidates(
    candidates: HashSet<usize>,
    query_tokens: &[String],
    index: &HashMap<String, HashSet<usize>>,
    idf: &HashMap<String, f64>,
) -> Vec<usize> {
    if candidates.len() <= 500 { return candidates.into_iter().collect(); }
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
}

pub fn search(query: &str, chunks: &[Chunk]) -> Vec<SearchResult> {
    if query.trim().is_empty() { return vec![]; }
    let (index, meta_list) = build_index(chunks);
    let n = chunks.len();
    let idf = compute_idf(&index, n);

    let query_tokens = tok(query);
    let query_symbols = extract_symbols(query);
    let query_lower = query.to_lowercase();

    let mut candidates: HashSet<usize> = HashSet::new();
    for t in &query_tokens { if let Some(s) = index.get(t) { candidates.extend(s); } }
    for s in &query_symbols { if let Some(ds) = index.get(s) { candidates.extend(ds); } }

    let scoring = prune_candidates(candidates, &query_tokens, &index, &idf);

    let mut scores: Vec<(usize, f64)> = Vec::new();
    for idx in scoring {
        let m = &meta_list[idx];
        let mut score = 0.0f64;
        if query_tokens.len() > 1 && m.content_lower.contains(&query_lower) { score += 30.0; }
        for sym in &query_symbols { if m.symbols.contains(sym) { score += 10.0; } }
        let file_matches = query_tokens.iter().filter(|t| m.file_name_tokens.contains(*t)).count();
        score += file_matches as f64 * 10.0;
        for t in &query_tokens {
            if let Some(ds) = index.get(t) {
                if ds.contains(&idx) {
                    let tf = (*m.frequency.get(t).unwrap_or(&1)).min(5) as f64;
                    let lb = if t.len() > 4 { 1.5 } else { 1.0 };
                    score += lb * tf * *idf.get(t).unwrap_or(&1.0);
                }
            }
        }
        if m.is_code { score *= 1.2; }
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
