use std::collections::HashMap;

pub const RRF_K: f64 = 60.0;
pub const IDENTIFIER_BOOST: f64 = 1.5;

pub fn looks_like_identifier(query: &str) -> bool {
    let q = query.trim();
    if q.is_empty() || q.contains(' ') { return false; }
    let has_separator = q.contains('_') || q.contains('-') || q.contains('.');
    let has_upper_lower = q.chars().any(|c| c.is_uppercase())
        && q.chars().any(|c| c.is_lowercase());
    has_separator || has_upper_lower
}

pub fn rrf_merge(
    bm25_ranked: &[String],
    vector_ranked: &[String],
    bm25_weight: f64,
    vector_weight: f64,
) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    for (rank, id) in bm25_ranked.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += bm25_weight / (RRF_K + (rank + 1) as f64);
    }
    for (rank, id) in vector_ranked.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += vector_weight / (RRF_K + (rank + 1) as f64);
    }
    let mut out: Vec<(String, f64)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

pub fn fuse(bm25_ranked: &[String], vector_ranked: &[String], query: &str) -> Vec<(String, f64)> {
    let bm25_w = if looks_like_identifier(query) { IDENTIFIER_BOOST } else { 1.0 };
    rrf_merge(bm25_ranked, vector_ranked, bm25_w, 1.0)
}

pub fn normalize_scores(scored: &[(String, f64)]) -> Vec<(String, f64)> {
    let max = scored.iter().map(|(_, s)| *s).fold(0f64, f64::max);
    if max <= 0.0 { return scored.to_vec(); }
    scored.iter().map(|(k, s)| (k.clone(), s / max)).collect()
}
