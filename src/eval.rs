use std::collections::HashMap;

pub fn dcg(rels: &[f64]) -> f64 {
    rels.iter().enumerate().map(|(i, r)| r / ((i + 2) as f64).log2()).sum()
}

pub fn ndcg_at_k(ranked_ids: &[String], qrels: &HashMap<String, f64>, k: usize) -> f64 {
    let topk: Vec<f64> = ranked_ids.iter().take(k)
        .map(|id| *qrels.get(id).unwrap_or(&0.0)).collect();
    let mut ideal: Vec<f64> = qrels.values().copied().collect();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    ideal.truncate(k);
    let idcg = dcg(&ideal);
    if idcg <= 0.0 { return 0.0; }
    dcg(&topk) / idcg
}

pub fn mrr(ranked_ids: &[String], qrels: &HashMap<String, f64>) -> f64 {
    for (i, id) in ranked_ids.iter().enumerate() {
        if qrels.get(id).map(|r| *r > 0.0).unwrap_or(false) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

pub fn recall_at_k(ranked_ids: &[String], qrels: &HashMap<String, f64>, k: usize) -> f64 {
    let total = qrels.values().filter(|r| **r > 0.0).count();
    if total == 0 { return 0.0; }
    let hit = ranked_ids.iter().take(k)
        .filter(|id| qrels.get(*id).map(|r| *r > 0.0).unwrap_or(false))
        .count();
    hit as f64 / total as f64
}

pub fn precision_at_k(ranked_ids: &[String], qrels: &HashMap<String, f64>, k: usize) -> f64 {
    if k == 0 { return 0.0; }
    let hit = ranked_ids.iter().take(k)
        .filter(|id| qrels.get(*id).map(|r| *r > 0.0).unwrap_or(false))
        .count();
    hit as f64 / k as f64
}

pub struct EvalReport {
    pub ndcg10: f64,
    pub mrr: f64,
    pub recall100: f64,
    pub p10: f64,
}

pub fn evaluate(ranked: &[String], qrels: &HashMap<String, f64>) -> EvalReport {
    EvalReport {
        ndcg10: ndcg_at_k(ranked, qrels, 10),
        mrr: mrr(ranked, qrels),
        recall100: recall_at_k(ranked, qrels, 100),
        p10: precision_at_k(ranked, qrels, 10),
    }
}

pub fn format_report(r: &EvalReport) -> String {
    format!(
        "NDCG@10: {:.4}\nMRR:     {:.4}\nRecall@100: {:.4}\nP@10:    {:.4}\n",
        r.ndcg10, r.mrr, r.recall100, r.p10
    )
}
