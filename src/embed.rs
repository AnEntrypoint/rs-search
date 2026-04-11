use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;
use std::sync::OnceLock;
use candle_core::{Device, DType, Tensor};
use candle_core::quantized::gguf_file;
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config, NomicBertModel, mean_pooling, l2_normalize};
use tokenizers::Tokenizer;
use crate::bm25::SearchResult;

struct Embedder {
    model: NomicBertModel,
    tokenizer: Tokenizer,
    device: Device,
}

static EMBEDDER: OnceLock<Result<Embedder, String>> = OnceLock::new();

fn load(model_path: &Path) -> Result<Embedder, String> {
    let device = Device::Cpu;
    let f = std::fs::File::open(model_path)
        .map_err(|e| format!("open model: {}", e))?;
    let mut reader = BufReader::new(f);
    let content = gguf_file::Content::read(&mut reader)
        .map_err(|e| format!("read gguf: {}", e))?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for name in content.tensor_infos.keys().cloned().collect::<Vec<_>>() {
        let qt = content.tensor(&mut reader, &name, &device)
            .map_err(|e| format!("tensor {}: {}", name, e))?;
        let t = qt.dequantize(&device)
            .map_err(|e| format!("dequantize {}: {}", name, e))?;
        tensors.insert(name, t);
    }
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let model = NomicBertModel::load(vb, &Config::default())
        .map_err(|e| format!("load model: {}", e))?;
    let tokenizer = Tokenizer::from_bytes(include_bytes!("../models/tokenizer.json"))
        .map_err(|e| format!("load tokenizer: {}", e))?;
    Ok(Embedder { model, tokenizer, device })
}

fn embed(embedder: &Embedder, text: &str) -> Result<Vec<f32>, String> {
    let enc = embedder.tokenizer.encode(text, true)
        .map_err(|e| format!("encode: {}", e))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    let mask: Vec<u32> = enc.get_attention_mask().to_vec();
    let len = ids.len();
    let input_ids = Tensor::from_vec(ids, (1, len), &embedder.device)
        .map_err(|e| format!("input_ids tensor: {}", e))?;
    let attn_mask = Tensor::from_vec(mask, (1, len), &embedder.device)
        .map_err(|e| format!("attn_mask tensor: {}", e))?;
    let hidden = embedder.model.forward(&input_ids, None, Some(&attn_mask))
        .map_err(|e| format!("forward: {}", e))?;
    let pooled = mean_pooling(&hidden, &attn_mask)
        .map_err(|e| format!("mean_pool: {}", e))?;
    let normed = l2_normalize(&pooled)
        .map_err(|e| format!("l2_norm: {}", e))?;
    normed.squeeze(0)
        .map_err(|e| format!("squeeze: {}", e))?
        .to_vec1::<f32>()
        .map_err(|e| format!("to_vec: {}", e))
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn embed_query(query: &str, model_path: &Path) -> Option<Vec<f32>> {
    let embedder = EMBEDDER.get_or_init(|| load(model_path));
    let embedder = embedder.as_ref().ok()?;
    let q_text = format!("search_query: {}", query);
    embed(embedder, &q_text).ok()
}

pub fn vector_search_texts(query: &str, items: &[(String, String)], model_path: &Path) -> Vec<(String, f32)> {
    let embedder = EMBEDDER.get_or_init(|| load(model_path));
    let embedder = match embedder {
        Ok(e) => e,
        Err(e) => { eprintln!("vector search unavailable: {}", e); return vec![]; }
    };
    let q_text = format!("search_query: {}", query);
    let q_emb = match embed(embedder, &q_text) {
        Ok(v) => v,
        Err(e) => { eprintln!("embed query: {}", e); return vec![]; }
    };
    let mut scored: Vec<(String, f32)> = items.iter().filter_map(|(id, text)| {
        let doc_text = format!("search_document: {}", &text[..text.len().min(1024)]);
        match embed(embedder, &doc_text) {
            Ok(d_emb) => Some((id.clone(), cosine(&q_emb, &d_emb))),
            Err(_) => None,
        }
    }).collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

pub fn rerank(mut results: Vec<SearchResult>, query: &str, model_path: &Path) -> Vec<SearchResult> {
    let embedder = EMBEDDER.get_or_init(|| load(model_path));
    let embedder = match embedder {
        Ok(e) => e,
        Err(e) => {
            eprintln!("vector search unavailable: {}", e);
            return results;
        }
    };
    let q_text = format!("search_query: {}", query);
    let q_emb = match embed(embedder, &q_text) {
        Ok(v) => v,
        Err(e) => { eprintln!("embed query: {}", e); return results; }
    };
    let max_bm25 = results.iter().map(|r| r.bm25_raw).fold(0f64, f64::max).max(1e-9);
    for r in &mut results {
        let doc_text = format!("search_document: {}", &r.chunk.content[..r.chunk.content.len().min(512)]);
        match embed(embedder, &doc_text) {
            Ok(d_emb) => {
                let sim = cosine(&q_emb, &d_emb);
                r.vector_score = Some(sim);
                let bm25_norm = (r.bm25_raw / max_bm25) as f32;
                let vec_norm = (sim + 1.0) / 2.0;
                r.score = (0.5 * bm25_norm + 0.5 * vec_norm) as f64;
            }
            Err(e) => eprintln!("embed doc: {}", e),
        }
    }
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}
