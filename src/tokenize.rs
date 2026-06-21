use std::collections::HashSet;

fn char_len(s: &str) -> usize {
    s.chars().count()
}

pub fn split_camel(word: &str) -> Vec<String> {
    let chars: Vec<char> = word.chars().collect();
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for i in 0..chars.len() {
        let c = chars[i];
        let is_upper = c.is_uppercase();
        let prev_lower = i > 0 && chars[i-1].is_lowercase();
        let next_lower = i + 1 < chars.len() && chars[i+1].is_lowercase();
        if is_upper && (prev_lower || next_lower) && !cur.is_empty() {
            if char_len(&cur) >= 1 { tokens.push(cur.to_lowercase()); }
            cur = c.to_string();
        } else {
            cur.push(c);
        }
    }
    if char_len(&cur) >= 1 { tokens.push(cur.to_lowercase()); }
    tokens
}

pub fn add_word_tokens(word: &str, out: &mut HashSet<String>) {
    if word != word.to_lowercase() {
        for t in split_camel(word) { if char_len(&t) >= 1 { out.insert(t); } }
    }
    for part in word.split(|c: char| c == '-' || c == '_' || c == '.') {
        let pc: String = part.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        if char_len(&pc) >= 1 { out.insert(pc); }
    }
    let cleaned: String = word.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect::<String>().to_lowercase();
    if char_len(&cleaned) >= 1 { out.insert(cleaned); }
}

pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: HashSet<String> = HashSet::new();
    for word in text.split_whitespace() { add_word_tokens(word, &mut tokens); }
    let mut v: Vec<String> = tokens.into_iter().collect();
    v.sort();
    v
}

#[cfg(test)]
mod eval_harness {
    use super::tokenize;
    use crate::eval::recall_at_k;
    use std::collections::HashMap;

    fn corpus() -> Vec<(&'static str, &'static str)> {
        vec![
            ("doc_afoo", "aFoo"),
            ("doc_xfn", "xField"),
            ("doc_ioerror", "IOError"),
            ("doc_parsejson", "parseJSON"),
            ("doc_a_b", "a_b_handler"),
            ("doc_multibyte", "naïveCase"),
            ("doc_httprequest", "HttpRequest"),
        ]
    }

    fn ranked_for_query(query: &str) -> Vec<String> {
        let q: Vec<String> = tokenize(query);
        let mut scored: Vec<(String, usize)> = corpus()
            .into_iter()
            .map(|(id, ident)| {
                let doc = tokenize(ident);
                let overlap = q.iter().filter(|t| doc.contains(t)).count();
                (id.to_string(), overlap)
            })
            .collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.into_iter().filter(|(_, n)| *n > 0).map(|(id, _)| id).collect()
    }

    fn qrels(relevant: &str) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert(relevant.to_string(), 1.0);
        m
    }

    #[test]
    fn single_char_segments_are_retrievable() {
        let ranked = ranked_for_query("a");
        assert!(recall_at_k(&ranked, &qrels("doc_afoo"), 10) > 0.0,
            "single-char segment 'a' from aFoo must be retrievable; got ranked={:?}", ranked);
        let ranked_x = ranked_for_query("x");
        assert!(recall_at_k(&ranked_x, &qrels("doc_xfn"), 10) > 0.0,
            "single-char segment 'x' from xField must be retrievable; got ranked={:?}", ranked_x);
    }

    #[test]
    fn multi_char_recall_not_regressed() {
        for (id, query) in [("doc_ioerror", "error"), ("doc_parsejson", "json"), ("doc_httprequest", "request")] {
            let ranked = ranked_for_query(query);
            assert!(recall_at_k(&ranked, &qrels(id), 10) > 0.0,
                "multi-char recall regressed for query '{}' expecting {}; got {:?}", query, id, ranked);
        }
    }

    #[test]
    fn multibyte_char_len_no_panic_and_tokenizes() {
        let toks = tokenize("naïveCase");
        assert!(toks.iter().any(|t| t.contains("ve") || t == "naïve" || t.contains("na")),
            "multibyte word must tokenize without byte-boundary loss; got {:?}", toks);
    }
}
