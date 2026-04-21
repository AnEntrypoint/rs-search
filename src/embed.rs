#[cfg(feature = "vector")]
use std::collections::HashMap;
#[cfg(feature = "vector")]
use std::io::Cursor;
use std::path::Path;
#[cfg(feature = "vector")]
use std::sync::OnceLock;
#[cfg(feature = "vector")]
use candle_core::{Device, DType, Tensor};
#[cfg(feature = "vector")]
use candle_core::quantized::gguf_file;
#[cfg(feature = "vector")]
use candle_nn::VarBuilder;
#[cfg(feature = "vector")]
use candle_transformers::models::nomic_bert::{Config, NomicBertModel, mean_pooling, l2_normalize};
#[cfg(feature = "vector")]
use tokenizers::Tokenizer;
use crate::bm25::SearchResult;
#[cfg(feature = "vector")]
use crate::embed_cache::EmbedCache;

#[cfg(feature = "vector")]
const MODEL_TAG: &str = "nomic-embed-text-v1.5";

#[cfg(feature = "vector")]
const MODEL_PARTS: [&[u8]; 6] = [
    include_bytes!("../models/nomic-embed-text-v1.5.Q4_K_M.gguf.part1"),
    include_bytes!("../models/nomic-embed-text-v1.5.Q4_K_M.gguf.part2"),
    include_bytes!("../models/nomic-embed-text-v1.5.Q4_K_M.gguf.part3"),
    include_bytes!("../models/nomic-embed-text-v1.5.Q4_K_M.gguf.part4"),
    include_bytes!("../models/nomic-embed-text-v1.5.Q4_K_M.gguf.part5"),
    include_bytes!("../models/nomic-embed-text-v1.5.Q4_K_M.gguf.part6"),
];

pub fn query_prefix() -> String {
    std::env::var("RS_SEARCH_QUERY_PREFIX")
        .unwrap_or_else(|_| "search_query: ".to_string())
}

pub fn doc_prefix() -> String {
    std::env::var("RS_SEARCH_DOC_PREFIX")
        .unwrap_or_else(|_| "search_document: ".to_string())
}

pub fn target_dim() -> Option<usize> {
    std::env::var("RS_SEARCH_DIM").ok().and_then(|s| s.parse::<usize>().ok())
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_feature = "sse")]
    {
        use simsimd::SpatialSimilarity;
        if let Some(d) = f32::cosine(a, b) { return 1.0 - d as f32; }
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na <= 0.0 || nb <= 0.0 { return 0.0; }
    dot / (na * nb)
}

pub fn truncate_mrl(vec: &[f32], dim: Option<usize>) -> Vec<f32> {
    let d = match dim { Some(d) if d > 0 && d < vec.len() => d, _ => return vec.to_vec() };
    let slice = &vec[..d];
    let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 0.0 { return slice.to_vec(); }
    slice.iter().map(|x| x / norm).collect()
}

#[cfg(feature = "vector")]
struct Embedder {
    model: NomicBertModel,
    tokenizer: Tokenizer,
    device: Device,
}

#[cfg(feature = "vector")]
static EMBEDDER: OnceLock<Result<Embedder, String>> = OnceLock::new();

#[cfg(feature = "vector")]
fn remap_gguf_to_hf(gguf_name: &str) -> Option<String> {
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        let dot = rest.find('.')?;
        let idx = &rest[..dot];
        let suffix = &rest[dot + 1..];
        let hf_suffix = match suffix {
            "attn_qkv.weight" => "attn.Wqkv.weight",
            "attn_output.weight" => "attn.out_proj.weight",
            "attn_output_norm.weight" => "norm1.weight",
            "attn_output_norm.bias" => "norm1.bias",
            "ffn_up.weight" => "mlp.fc11.weight",
            "ffn_gate.weight" => "mlp.fc12.weight",
            "ffn_down.weight" => "mlp.fc2.weight",
            "layer_output_norm.weight" => "norm2.weight",
            "layer_output_norm.bias" => "norm2.bias",
            _ => return None,
        };
        return Some(format!("encoder.layers.{}.{}", idx, hf_suffix));
    }
    Some(match gguf_name {
        "token_embd.weight" => "embeddings.word_embeddings.weight".to_string(),
        "token_types.weight" => "embeddings.token_type_embeddings.weight".to_string(),
        "token_embd_norm.weight" => "emb_ln.weight".to_string(),
        "token_embd_norm.bias" => "emb_ln.bias".to_string(),
        _ => return None,
    })
}

#[cfg(feature = "vector")]
fn load() -> Result<Embedder, String> {
    let device = Device::Cpu;
    let total: usize = MODEL_PARTS.iter().map(|p| p.len()).sum();
    let mut blob = Vec::with_capacity(total);
    for p in MODEL_PARTS { blob.extend_from_slice(p); }
    let mut reader = Cursor::new(blob);
    let content = gguf_file::Content::read(&mut reader).map_err(|e| format!("read gguf: {}", e))?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for name in content.tensor_infos.keys().cloned().collect::<Vec<_>>() {
        let qt = content.tensor(&mut reader, &name, &device).map_err(|e| format!("tensor {}: {}", name, e))?;
        let t = qt.dequantize(&device).map_err(|e| format!("dequantize {}: {}", name, e))?;
        let hf = remap_gguf_to_hf(&name)
            .ok_or_else(|| format!("unmapped gguf tensor: {}", name))?;
        tensors.insert(hf, t);
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let cfg = Config { vocab_size: 30522, ..Config::default() };
    let model = NomicBertModel::load(vb, &cfg).map_err(|e| format!("load model: {}", e))?;
    let tokenizer = Tokenizer::from_bytes(include_bytes!("../models/tokenizer.json"))
        .map_err(|e| format!("load tokenizer: {}", e))?;
    Ok(Embedder { model, tokenizer, device })
}

#[cfg(feature = "vector")]
fn embed_raw(embedder: &Embedder, text: &str) -> Result<Vec<f32>, String> {
    let enc = embedder.tokenizer.encode(text, true).map_err(|e| format!("encode: {}", e))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let mask: Vec<u32> = enc.get_attention_mask().to_vec();
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), &embedder.device).map_err(|e| format!("input_ids: {}", e))?;
    let attn_mask = Tensor::from_vec(mask, (1, len), &embedder.device).map_err(|e| format!("attn_mask: {}", e))?;
    let hidden = embedder.model.forward(&input_ids, None, Some(&attn_mask)).map_err(|e| format!("forward: {}", e))?;
    let pooled = mean_pooling(&hidden, &attn_mask).map_err(|e| format!("mean_pool: {}", e))?;
    let normed = l2_normalize(&pooled).map_err(|e| format!("l2_norm: {}", e))?;
    normed.squeeze(0).map_err(|e| format!("squeeze: {}", e))?
        .to_vec1::<f32>().map_err(|e| format!("to_vec: {}", e))
}

#[cfg(feature = "vector")]
fn embed_with_cache(embedder: &Embedder, text: &str, cache: Option<&EmbedCache>, dim: Option<usize>) -> Result<Vec<f32>, String> {
    let effective_dim = dim.unwrap_or(0);
    if let Some(c) = cache {
        let key = EmbedCache::key(MODEL_TAG, effective_dim, text);
        if let Some(v) = c.get(&key) { return Ok(v); }
        let raw = embed_raw(embedder, text)?;
        let out = truncate_mrl(&raw, dim);
        c.put(&key, &out);
        return Ok(out);
    }
    let raw = embed_raw(embedder, text)?;
    Ok(truncate_mrl(&raw, dim))
}

#[cfg(feature = "vector")]
fn cache_for(root: &Path) -> Option<EmbedCache> {
    let p = root.join(".code-search");
    if !p.exists() { let _ = std::fs::create_dir_all(&p); }
    Some(EmbedCache::new(&p))
}

#[cfg(feature = "vector")]
pub fn embed_query(query: &str, _model_path: &Path) -> Option<Vec<f32>> {
    let embedder = EMBEDDER.get_or_init(load).as_ref().ok()?;
    let text = format!("{}{}", query_prefix(), query);
    embed_with_cache(embedder, &text, None, target_dim()).ok()
}

#[cfg(not(feature = "vector"))]
pub fn embed_query(_query: &str, _model_path: &Path) -> Option<Vec<f32>> { None }

#[cfg(feature = "vector")]
pub fn vector_search_texts(query: &str, items: &[(String, String)], _model_path: &Path) -> Vec<(String, f32)> {
    let embedder = EMBEDDER.get_or_init(load).as_ref().expect("embedded GGUF must load — rs-search binary is broken if this fails");
    let dim = target_dim();
    let cache = cache_for(&std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf()));
    let q_text = format!("{}{}", query_prefix(), query);
    let q_emb = embed_with_cache(embedder, &q_text, cache.as_ref(), dim).expect("embed query");
    let mut scored: Vec<(String, f32)> = items.iter().filter_map(|(id, text)| {
        let doc_text = format!("{}{}", doc_prefix(), &text[..text.len().min(1024)]);
        embed_with_cache(embedder, &doc_text, cache.as_ref(), dim).ok()
            .map(|d| (id.clone(), cosine(&q_emb, &d)))
    }).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

#[cfg(not(feature = "vector"))]
pub fn vector_search_texts(_query: &str, _items: &[(String, String)], _model_path: &Path) -> Vec<(String, f32)> { vec![] }

#[cfg(feature = "vector")]
pub fn rerank(mut results: Vec<SearchResult>, query: &str, _model_path: &Path) -> Vec<SearchResult> {
    let embedder = EMBEDDER.get_or_init(load).as_ref().expect("embedded GGUF must load — rs-search binary is broken if this fails");
    let dim = target_dim();
    let cache = cache_for(&std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf()));
    let q_text = format!("{}{}", query_prefix(), query);
    let q_emb = embed_with_cache(embedder, &q_text, cache.as_ref(), dim).expect("embed query");
    let bm25_ranked: Vec<String> = results.iter().enumerate().map(|(i, _)| i.to_string()).collect();
    let mut vec_scores: Vec<(usize, f32)> = Vec::with_capacity(results.len());
    for (i, r) in results.iter().enumerate() {
        let doc_text = format!("{}{}", doc_prefix(), &r.chunk.content[..r.chunk.content.len().min(1024)]);
        if let Ok(d) = embed_with_cache(embedder, &doc_text, cache.as_ref(), dim) {
            vec_scores.push((i, cosine(&q_emb, &d)));
        }
    }
    for &(i, s) in &vec_scores { results[i].vector_score = Some(s); }
    let mut sorted = vec_scores.clone();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let vec_ranked: Vec<String> = sorted.iter().map(|(i, _)| i.to_string()).collect();
    let fused = crate::fusion::fuse(&bm25_ranked, &vec_ranked, query);
    let fused = crate::fusion::normalize_scores(&fused);
    let order: HashMap<String, f64> = fused.iter().cloned().collect();
    for (i, r) in results.iter_mut().enumerate() {
        r.score = *order.get(&i.to_string()).unwrap_or(&0.0);
    }
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(not(feature = "vector"))]
pub fn rerank(results: Vec<SearchResult>, _query: &str, _model_path: &Path) -> Vec<SearchResult> { results }
