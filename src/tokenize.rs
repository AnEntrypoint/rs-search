use std::collections::HashSet;

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
            if cur.len() > 1 { tokens.push(cur.to_lowercase()); }
            cur = c.to_string();
        } else {
            cur.push(c);
        }
    }
    if cur.len() > 1 { tokens.push(cur.to_lowercase()); }
    tokens
}

pub fn add_word_tokens(word: &str, out: &mut HashSet<String>) {
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
