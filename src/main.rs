use rs_search::{bm25, context, embed, explain, git, mcp, mtime_cache, scanner};
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
    let vector_results = embed::rerank(bm25_results.clone(), query, Path::new(""));

    println!("\n=== BM25 RESULTS ===");
    print_code_results(&bm25_results, root);
    println!("\n=== VECTOR RESULTS ===");
    print_code_results(&vector_results, root);

    if is_git {
        println!("\nIndexing git commits...");
        let commits = git::scan_git_commits(root, 200);
        println!("Found {} commits with indexed-file diffs\n", commits.len());
        let commit_texts = git::commits_to_searchable(&commits);
        println!("=== MOST RELEVANT COMMITS (BM25) ===");
        let bm25_commits = bm25::search_texts(query, &commit_texts);
        for (i, (hash, score)) in bm25_commits.iter().take(10).enumerate() {
            println!("{}. {} (score: {:.0}%)", i + 1, &hash[..hash.len().min(12)], score * 100.0);
        }
        if bm25_commits.is_empty() { println!("  (no results)"); }
        println!("\n=== MOST RELEVANT COMMITS (vector) ===");
        let vec_commits = embed::vector_search_texts(query, &commit_texts, Path::new(""));
        for (i, (hash, score)) in vec_commits.iter().take(10).enumerate() {
            println!("{}. {} (score: {:.0}%)", i + 1, &hash[..hash.len().min(12)], ((*score + 1.0) / 2.0) * 100.0);
        }
        if vec_commits.is_empty() { println!("  (no results)"); }
    }
}

fn update_mtime_cache(chunks: &[scanner::Chunk], cache: &mut mtime_cache::MtimeCache) {
    let mut scanned = std::collections::HashMap::new();
    for c in chunks { scanned.insert(c.file_path.clone(), c.mtime); }
    for (fp, &mt) in &scanned { cache.insert(fp.clone(), mt); }
    cache.save();
}

fn print_code_results(results: &[bm25::SearchResult], root: &Path) {
    if results.is_empty() { println!("No results found."); return; }
    println!("Found {} result{}:\n", results.len(), if results.len() != 1 { "s" } else { "" });
    for (i, r) in results.iter().enumerate() {
        let score_pct = format!("{:.1}", r.score * 100.0);
        let total = context::get_file_total_lines(root, &r.chunk.file_path).map(|n| format!(" [{}L]", n)).unwrap_or_default();
        let ctx = context::find_enclosing_context(&r.chunk.content, r.chunk.line_start).map(|c| format!(" (in: {})", c)).unwrap_or_default();
        println!("{}. {}{}: {}-{}{} (score: {}%)", i + 1, r.chunk.file_path, total, r.chunk.line_start, r.chunk.line_end, ctx, score_pct);
        println!("   BM25: {:.2}", r.bm25_raw);
        if let Some(vs) = r.vector_score { println!("   Vector: {:.4}", vs); }
        for line in r.chunk.content.split('\n').take(3) {
            println!("   > {}", &line[..line.len().min(80)]);
        }
        println!();
    }
}
