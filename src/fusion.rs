use std::collections::HashMap;

pub const RRF_K: f64 = 60.0;
pub const IDENTIFIER_BOOST: f64 = 1.5;

pub fn looks_like_identifier(query: &str) -> bool {
    let q = query.trim();
    if q.is_empty() || q.contains(' ') { return false; }
    let has_non_digit_non_period = q.chars().any(|c| c != '.' && !c.is_ascii_digit());
    let has_separator = has_non_digit_non_period
        && (q.contains('_') || q.contains('-') || q.contains('.'));
    let has_upper_lower = q.chars().any(|c| c.is_uppercase())
        && q.chars().any(|c| c.is_lowercase());
    has_separator || has_upper_lower
}

pub fn rrf_merge_n(ranked_lists: &[Vec<String>]) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    for ranked in ranked_lists {
        for (rank, id) in ranked.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (RRF_K + (rank + 1) as f64);
        }
    }
    let mut out: Vec<(String, f64)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    out
}

pub fn rrf_merge_n_weighted(ranked_lists: &[Vec<String>], weights: &[f64]) -> Vec<(String, f64)> {
    let mut scores: HashMap<String, f64> = HashMap::new();
    for (li, ranked) in ranked_lists.iter().enumerate() {
        let w = weights.get(li).copied().unwrap_or(1.0);
        for (rank, id) in ranked.iter().enumerate() {
            *scores.entry(id.clone()).or_insert(0.0) += w / (RRF_K + (rank + 1) as f64);
        }
    }
    let mut out: Vec<(String, f64)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.0.cmp(&b.0)));
    out
}

pub fn fuse_n(ranked_lists: &[Vec<String>], weights: &[f64], query: &str) -> Vec<(String, f64)> {
    if looks_like_identifier(query) {
        rrf_merge_n_weighted(ranked_lists, weights)
    } else {
        rrf_merge_n(ranked_lists)
    }
}

#[cfg(test)]
mod fusion_eval {
    use super::{fuse_n, rrf_merge_n, IDENTIFIER_BOOST};
    use crate::eval::{ndcg_at_k, mrr};
    use std::collections::HashMap;

    fn qrels(ids: &[(&str, f64)]) -> HashMap<String, f64> {
        ids.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn identifier_query_boosts_exact_token_hit() {
        let vector = vec!["doc_semantic".to_string(), "doc_exact".to_string()];
        let bm25 = vec!["doc_exact".to_string(), "doc_semantic".to_string()];
        let git: Vec<String> = vec![];
        let lists = vec![vector, bm25, git];
        let weights = [1.0, IDENTIFIER_BOOST, 1.0];
        let rel = qrels(&[("doc_exact", 1.0)]);

        let boosted: Vec<String> = fuse_n(&lists, &weights, "parseConfig").into_iter().map(|(id, _)| id).collect();
        let equal: Vec<String> = rrf_merge_n(&lists).into_iter().map(|(id, _)| id).collect();

        assert!(mrr(&boosted, &rel) >= mrr(&equal, &rel),
            "identifier-boost must not lower MRR of the exact-token hit; boosted={:?} equal={:?}", boosted, equal);
        assert_eq!(boosted.first().map(|s| s.as_str()), Some("doc_exact"),
            "for an identifier query, the exact-token (bm25-top) hit should rank first; got {:?}", boosted);
    }

    #[test]
    fn prose_query_uses_equal_weight() {
        let vector = vec!["doc_a".to_string(), "doc_b".to_string()];
        let bm25 = vec!["doc_b".to_string(), "doc_a".to_string()];
        let lists = vec![vector, bm25, vec![]];
        let weights = [1.0, IDENTIFIER_BOOST, 1.0];
        let prose: Vec<String> = fuse_n(&lists, &weights, "how to parse the config").into_iter().map(|(id, _)| id).collect();
        let equal: Vec<String> = rrf_merge_n(&lists).into_iter().map(|(id, _)| id).collect();
        assert_eq!(prose, equal, "a multi-word prose query must use equal-weight fusion (no identifier boost)");
    }

    #[test]
    fn ndcg_not_regressed_for_identifier_query() {
        let vector = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        let bm25 = vec!["y".to_string(), "z".to_string(), "x".to_string()];
        let lists = vec![vector, bm25, vec![]];
        let weights = [1.0, IDENTIFIER_BOOST, 1.0];
        let rel = qrels(&[("y", 1.0), ("z", 0.5)]);
        let boosted: Vec<String> = fuse_n(&lists, &weights, "yParser").into_iter().map(|(id, _)| id).collect();
        let equal: Vec<String> = rrf_merge_n(&lists).into_iter().map(|(id, _)| id).collect();
        assert!(ndcg_at_k(&boosted, &rel, 10) >= ndcg_at_k(&equal, &rel, 10) - 1e-9,
            "identifier boost must not reduce NDCG@10; boosted={:?} equal={:?}", boosted, equal);
    }
}
