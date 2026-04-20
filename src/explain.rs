use std::collections::{HashMap, HashSet};
use std::path::Path;
use crate::bm25::{tokenize, SearchResult, search};
use crate::scanner::{Chunk, scan_repository};
use crate::fusion::looks_like_identifier;

pub struct TokenBreakdown {
    pub token: String,
    pub idf: f64,
    pub doc_freq: usize,
}

pub struct ChunkExplanation {
    pub chunk: Chunk,
    pub final_score: f64,
    pub bm25_raw: f64,
    pub vector_score: Option<f32>,
    pub matched_tokens: Vec<String>,
}

pub struct Explanation {
    pub query: String,
    pub is_identifier_query: bool,
    pub query_tokens: Vec<String>,
    pub token_breakdown: Vec<TokenBreakdown>,
    pub results: Vec<ChunkExplanation>,
}

fn compute_idf_table(chunks: &[Chunk], query_tokens: &[String]) -> Vec<TokenBreakdown> {
    let n = chunks.len();
    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    for token in query_tokens {
        let count = chunks.iter().filter(|c| {
            let tokens: HashSet<String> = tokenize(&c.content).into_iter().collect();
            tokens.contains(token)
        }).count();
        doc_freq.insert(token.clone(), count);
    }
    query_tokens.iter().map(|t| {
        let df = *doc_freq.get(t).unwrap_or(&0);
        let idf = ((n + 1) as f64 / (df + 1) as f64).ln() + 1.0;
        TokenBreakdown { token: t.clone(), idf, doc_freq: df }
    }).collect()
}

fn matched_tokens_for(chunk: &Chunk, query_tokens: &[String]) -> Vec<String> {
    let content_tokens: HashSet<String> = tokenize(&chunk.content).into_iter().collect();
    query_tokens.iter().filter(|t| content_tokens.contains(*t)).cloned().collect()
}

pub fn explain(query: &str, root: &Path) -> Explanation {
    let chunks = scan_repository(root);
    let results: Vec<SearchResult> = search(query, &chunks);
    let query_tokens = tokenize(query);
    let token_breakdown = compute_idf_table(&chunks, &query_tokens);
    let chunk_explanations = results.into_iter().map(|r| ChunkExplanation {
        matched_tokens: matched_tokens_for(&r.chunk, &query_tokens),
        chunk: r.chunk,
        final_score: r.score,
        bm25_raw: r.bm25_raw,
        vector_score: r.vector_score,
    }).collect();
    Explanation {
        query: query.to_string(),
        is_identifier_query: looks_like_identifier(query),
        query_tokens,
        token_breakdown,
        results: chunk_explanations,
    }
}

pub fn format(e: &Explanation) -> String {
    let mut out = String::new();
    out.push_str(&format!("Query: \"{}\"\n", e.query));
    out.push_str(&format!("Identifier-like: {}\n", e.is_identifier_query));
    out.push_str(&format!("Tokens: {:?}\n\n", e.query_tokens));
    out.push_str("Token IDF breakdown:\n");
    for t in &e.token_breakdown {
        out.push_str(&format!("  {:<20} df={:<5} idf={:.4}\n", t.token, t.doc_freq, t.idf));
    }
    out.push_str(&format!("\nFusion weights: bm25={:.2} vector=1.00 (RRF k=60)\n\n",
        if e.is_identifier_query { 1.5 } else { 1.0 }));
    out.push_str("Top results:\n");
    for (i, r) in e.results.iter().enumerate() {
        out.push_str(&format!("\n{}. {}: {}-{}\n", i + 1, r.chunk.file_path, r.chunk.line_start, r.chunk.line_end));
        out.push_str(&format!("   final_score={:.4}  bm25_raw={:.2}", r.final_score, r.bm25_raw));
        if let Some(v) = r.vector_score { out.push_str(&format!("  vector={:.4}", v)); }
        out.push('\n');
        out.push_str(&format!("   matched tokens: {:?}\n", r.matched_tokens));
    }
    out
}
