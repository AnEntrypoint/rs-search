use rs_search::{bm25, context, embed, explain, git, mcp, mtime_cache, scanner, tokenize};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use clap::{Parser, Subcommand};

#[cfg(feature = "perf")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn build_features() -> String {
    let mut feats: Vec<&str> = Vec::new();
    if cfg!(feature = "vector") { feats.push("vector"); }
    if cfg!(feature = "perf") { feats.push("perf"); }
    if cfg!(feature = "pdf") { feats.push("pdf"); }
    if feats.is_empty() { "none".to_string() } else { feats.join(", ") }
}

#[derive(Parser)]
#[command(
    name = "rs-search",
    version,
    about = "BM25 + vector codebase search with MCP protocol support",
    after_help = "Run `rs-search --features` to see enabled build features."
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[arg(long, help = "Print enabled build features and exit")]
    features: bool,
    query: Vec<String>,
}

#[derive(Subcommand)]
enum Command {
    Serve,
    Explain { query: Vec<String> },
    Search { query: Vec<String> },
}

fn main() {
    let cli = Cli::parse();

    if cli.features {
        println!("rs-search {} features: {}", env!("CARGO_PKG_VERSION"), build_features());
        return;
    }

    let root = std::env::current_dir().expect("cwd");

    match cli.command {
        Some(Command::Serve) => { mcp::run_mcp_server(); return; }
        Some(Command::Explain { query }) => {
            let q = query.join(" ");
            let e = explain::explain(&q, &root);
            print!("{}", explain::format(&e));
            return;
        }
        Some(Command::Search { query }) => {
            run_full_search(&query.join(" "), &root);
            return;
        }
        None => {}
    }

    if cli.query.is_empty() {
        mcp::run_mcp_server();
        return;
    }
    run_full_search(&cli.query.join(" "), &root);
}

fn run_full_search(query: &str, root: &Path) {
    println!("Code Search Tool\nRoot: {}\n", root.display());
    let is_git = root.join(".git").exists();
    if !is_git { eprintln!("Warning: Not a git repository. Indexing current directory anyway.\n"); }

    let db_path = root.join(".gm").join("code-search");
    let legacy = root.join(".code-search");
    if legacy.exists() && !db_path.exists() {
        let _ = fs::create_dir_all(root.join(".gm"));
        let _ = fs::rename(&legacy, &db_path);
    }
    let _ = fs::create_dir_all(&db_path);

    let mut cache = mtime_cache::MtimeCache::load(&db_path);
    println!("Scanning repository...");
    let chunks = scanner::scan_repository(root);
    println!("Found {} code chunks\n", chunks.len());

    update_mtime_cache(&chunks, &mut cache);

    let bm25_results = bm25::search(query, &chunks);
    println!("Applying vector re-ranking...");
    let vector_results = embed::rerank(bm25_results.clone(), query, &db_path);

    let query_tokens = tokenize::tokenize(query);
    println!("\n=== BM25 RESULTS ===");
    print_code_results(&bm25_results, root, &query_tokens);
    println!("\n=== VECTOR RESULTS ===");
    print_code_results(&vector_results, root, &query_tokens);

    sweep_emb_cache(&chunks, query, &db_path);

    if is_git {
        println!("\nIndexing git commits...");
        let commits = git::scan_git_commits(root, 200);
        println!("Found {} commits with indexed-file diffs\n", commits.len());
        let commit_texts = git::commits_to_searchable(&commits);
        let commit_lookup: std::collections::HashMap<String, &git::CommitInfo> =
            commits.iter().map(|c| (c.hash.clone(), c)).collect();
        println!("=== MOST RELEVANT COMMITS (BM25) ===");
        let bm25_commits: Vec<(String, f32)> = bm25::search_texts(query, &commit_texts)
            .into_iter().map(|(s, v)| (s, v as f32)).collect();
        print_commit_results(&bm25_commits, &commit_lookup, false);
        println!("\n=== MOST RELEVANT COMMITS (vector) ===");
        let vec_commits = embed::vector_search_texts(query, &commit_texts, &db_path);
        print_commit_results(&vec_commits, &commit_lookup, true);
    }
}

fn print_commit_results(
    ranked: &[(String, f32)],
    commits: &std::collections::HashMap<String, &git::CommitInfo>,
    cosine: bool,
) {
    if ranked.is_empty() { println!("  (no results)"); return; }
    for (i, (hash, score)) in ranked.iter().take(10).enumerate() {
        let pct = if cosine { ((*score + 1.0) / 2.0) * 100.0 } else { *score * 100.0 };
        let short = &hash[..hash.len().min(12)];
        let info = commits.get(hash);
        let subject = info
            .and_then(|c| c.message.lines().next().map(|s| s.to_string()))
            .unwrap_or_default();
        let (files, plus, minus) = info.map(|c| diff_stats(&c.diff_text)).unwrap_or((0, 0, 0));
        let raw_label = if cosine { "cos" } else { "raw" };
        println!(
            "{:>2}. {}  {:>5.1}%  ({}={:.4})  +{}/-{} across {} file{}",
            i + 1, short, pct, raw_label, score, plus, minus, files, if files == 1 { "" } else { "s" }
        );
        if !subject.is_empty() { println!("    {}", truncate_str(&subject, 100)); }
    }
}

fn diff_stats(diff: &str) -> (usize, usize, usize) {
    let mut files = 0usize;
    let mut plus = 0usize;
    let mut minus = 0usize;
    for line in diff.split('\n') {
        if line.starts_with("diff --git") { files += 1; }
        else if line.starts_with('+') && !line.starts_with("+++ ") { plus += 1; }
        else if line.starts_with('-') && !line.starts_with("--- ") { minus += 1; }
    }
    (files, plus, minus)
}

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max { return s.to_string(); }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

fn sweep_emb_cache(chunks: &[scanner::Chunk], query: &str, db_path: &Path) {
    let dim = embed::target_dim().unwrap_or(0);
    let model_tag = "nomic-embed-text-v1.5";
    let mut live: HashSet<String> = HashSet::new();
    let qp = embed::query_prefix();
    let dp = embed::doc_prefix();
    live.insert(rs_search::embed_cache::EmbedCache::key(model_tag, dim, &format!("{}{}", qp, query)));
    for c in chunks {
        live.insert(rs_search::embed_cache::EmbedCache::key(
            model_tag, dim, &format!("{}{}", dp, c.content),
        ));
    }
    let cache = rs_search::embed_cache::EmbedCache::new(db_path);
    let (removed, freed) = cache.sweep_orphans(&live);
    if removed > 0 {
        println!("emb-cache: swept {} orphaned vectors ({} KB freed)", removed, freed / 1024);
    }
}

fn update_mtime_cache(chunks: &[scanner::Chunk], cache: &mut mtime_cache::MtimeCache) {
    let mut scanned = std::collections::HashMap::new();
    for c in chunks { scanned.insert(c.file_path.clone(), c.mtime); }
    for (fp, &mt) in &scanned { cache.insert(fp.clone(), mt); }
    cache.save();
}

fn print_code_results(results: &[bm25::SearchResult], root: &Path, query_tokens: &[String]) {
    if results.is_empty() { println!("No results found."); return; }
    println!("Found {} result{}:\n", results.len(), if results.len() != 1 { "s" } else { "" });
    let max_score = results.iter().map(|r| r.score).fold(0f64, f64::max).max(1e-9);
    let q_set: HashSet<&str> = query_tokens.iter().map(|s| s.as_str()).collect();
    for (i, r) in results.iter().enumerate() {
        let score_pct = format!("{:.1}", r.score * 100.0);
        let conf_pct = (r.score / max_score) * 100.0;
        let confidence_band = match conf_pct {
            p if p >= 80.0 => "strong",
            p if p >= 50.0 => "good",
            p if p >= 25.0 => "weak",
            _ => "marginal",
        };
        let total = context::get_file_total_lines(root, &r.chunk.file_path).map(|n| format!(" [{}L]", n)).unwrap_or_default();
        let ctx = context::find_enclosing_context(&r.chunk.content, r.chunk.line_start).map(|c| format!(" (in: {})", c)).unwrap_or_default();
        let chunk_tokens: HashSet<String> = tokenize::tokenize(&r.chunk.content).into_iter().collect();
        let matched: Vec<&str> = q_set.iter().copied().filter(|t| chunk_tokens.contains(*t)).collect();
        let missing: Vec<&str> = q_set.iter().copied().filter(|t| !chunk_tokens.contains(*t)).collect();
        let coverage = if q_set.is_empty() { 0.0 } else { matched.len() as f64 / q_set.len() as f64 * 100.0 };
        println!("{}. {}{}: {}-{}{} (score: {}% — {} {:.0}% of best)", i + 1, r.chunk.file_path, total, r.chunk.line_start, r.chunk.line_end, ctx, score_pct, confidence_band, conf_pct);
        print!("   bm25={:.2}", r.bm25_raw);
        if let Some(vs) = r.vector_score { print!("  vector={:.4}", vs); }
        println!("  fused={:.4}", r.score);
        println!("   token coverage: {:.0}% ({}/{})", coverage, matched.len(), q_set.len());
        if !matched.is_empty() { println!("   matched: {:?}", matched); }
        if !missing.is_empty() { println!("   missing: {:?}", missing); }
        for line in r.chunk.content.split('\n').take(3) {
            println!("   > {}", &line[..line.len().min(80)]);
        }
        println!();
    }
}
