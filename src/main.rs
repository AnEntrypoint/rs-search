use rs_search::{assemble, bm25, context, embed, git, mcp, mtime_cache, scanner};
use std::fs;
use std::path::Path;
use clap::Parser;

#[derive(Parser)]
#[command(name = "rs-search", about = "BM25 + vector codebase search with MCP protocol support")]
struct Cli {
    query: Vec<String>,
}

fn ensure_gitignore_entry(root: &Path) {
    let gi = root.join(".gitignore");
    let entry = ".code-search/";
    if gi.exists() {
        let content = fs::read_to_string(&gi).unwrap_or_default();
        if !content.contains(entry) {
            let _ = fs::write(&gi, format!("{}\n{}", content.trim_end(), entry));
        }
    } else {
        let _ = fs::write(&gi, format!("{}\n", entry));
    }
}

fn main() {
    let cli = Cli::parse();

    if cli.query.is_empty() {
        mcp::run_mcp_server();
        return;
    }

    let query = cli.query.join(" ");
    let root = std::env::current_dir().expect("cwd");

    println!("Code Search Tool");
    println!("Root: {}\n", root.display());

    let is_git = root.join(".git").exists();
    if !is_git { eprintln!("Warning: Not a git repository. Indexing current directory anyway.\n"); }

    ensure_gitignore_entry(&root);

    let db_path = root.join(".code-search");
    let _ = fs::create_dir_all(&db_path);

    let models_dir = db_path.join("models");
    let _ = fs::create_dir_all(&models_dir);

    let exe_dir = std::env::current_exe().ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    let src_models = exe_dir.as_deref()
        .map(|d| d.join("models"))
        .filter(|p| p.exists())
        .or_else(|| {
            let m = root.join("models");
            if m.exists() { Some(m) } else { None }
        });

    let model_path = if !assemble::model_path(&models_dir).exists() {
        if let Some(src) = &src_models {
            match assemble::ensure_assembled(src) {
                Ok(p) => {
                    let dst = models_dir.join(p.file_name().unwrap());
                    if !dst.exists() { let _ = fs::copy(&p, &dst); }
                    Some(dst)
                }
                Err(e) => { eprintln!("model assemble: {}", e); None }
            }
        } else { None }
    } else {
        Some(assemble::model_path(&models_dir))
    };

    let mut cache = mtime_cache::MtimeCache::load(&db_path);

    println!("Scanning repository...");
    let chunks = scanner::scan_repository(&root);
    println!("Found {} code chunks\n", chunks.len());

    let mut scanned_mtimes = std::collections::HashMap::new();
    for c in &chunks { scanned_mtimes.insert(c.file_path.clone(), c.mtime); }

    let mut files_to_reindex = std::collections::HashSet::new();
    for (fp, &mtime) in &scanned_mtimes {
        if cache.get(fp).map(|m| m != mtime).unwrap_or(true) {
            files_to_reindex.insert(fp.clone());
        }
    }

    let deleted: Vec<String> = {
        let scanned_set: std::collections::HashSet<_> = scanned_mtimes.keys().cloned().collect();
        chunks.iter().map(|c| c.file_path.clone())
            .collect::<std::collections::HashSet<_>>()
            .difference(&scanned_set).cloned().collect()
    };

    for fp in &deleted { cache.remove(fp); }
    for fp in &files_to_reindex {
        if let Some(&mt) = scanned_mtimes.get(fp) { cache.insert(fp.clone(), mt); }
    }
    cache.save();

    println!("Files to re-index: {} ({} chunks), deleted: {}\n",
        files_to_reindex.len(),
        chunks.iter().filter(|c| files_to_reindex.contains(&c.file_path)).count(),
        deleted.len());

    let bm25_results = bm25::search(&query, &chunks);
    let model_exists = model_path.as_ref().map(|p| p.exists()).unwrap_or(false);
    let vector_results = if let Some(mp) = &model_path {
        println!("Applying vector re-ranking...");
        embed::rerank(bm25_results.clone(), &query, mp)
    } else {
        eprintln!("Vector model not available, using BM25 only.");
        bm25_results.clone()
    };

    println!("\n=== BM25 RESULTS ===");
    print_code_results(&bm25_results, &root);

    println!("\n=== VECTOR RESULTS ===");
    print_code_results(&vector_results, &root);

    if is_git {
        println!("\nIndexing git commits...");
        let commits = git::scan_git_commits(&root, 200);
        println!("Found {} commits with indexed-file diffs\n", commits.len());
        let commit_texts = git::commits_to_searchable(&commits);

        println!("=== MOST RELEVANT COMMITS (BM25) ===");
        let bm25_commits = bm25::search_texts(&query, &commit_texts);
        for (i, (hash, score)) in bm25_commits.iter().take(10).enumerate() {
            println!("{}. {} (score: {:.0}%)", i + 1, &hash[..hash.len().min(12)], score * 100.0);
        }
        if bm25_commits.is_empty() { println!("  (no results)"); }

        if model_exists {
            if let Some(mp) = &model_path {
                println!("\n=== MOST RELEVANT COMMITS (vector) ===");
                let vec_commits = embed::vector_search_texts(&query, &commit_texts, mp);
                for (i, (hash, score)) in vec_commits.iter().take(10).enumerate() {
                    println!("{}. {} (score: {:.0}%)", i + 1, &hash[..hash.len().min(12)], ((*score + 1.0) / 2.0) * 100.0);
                }
                if vec_commits.is_empty() { println!("  (no results)"); }
            }
        }
    }
}

fn print_code_results(results: &[bm25::SearchResult], root: &Path) {
    if results.is_empty() { println!("No results found."); return; }
    println!("Found {} result{}:\n", results.len(), if results.len() != 1 { "s" } else { "" });
    for (i, r) in results.iter().enumerate() {
        let score_pct = format!("{:.1}", r.score * 100.0);
        let abs_path = root.join(r.chunk.file_path.replace('/', std::path::MAIN_SEPARATOR_STR));
        let total = context::get_file_total_lines(root, &r.chunk.file_path).map(|n| format!(" [{}L]", n)).unwrap_or_default();
        let ctx = context::find_enclosing_context(&r.chunk.content, r.chunk.line_start)
            .map(|c| format!(" (in: {})", c)).unwrap_or_default();
        println!("{}. {}{}: {}-{}{} (score: {}%)", i + 1, r.chunk.file_path, total, r.chunk.line_start, r.chunk.line_end, ctx, score_pct);
        println!("   BM25: {:.2}", r.bm25_raw);
        if let Some(vs) = r.vector_score { println!("   Vector: {:.4}", vs); }
        for line in r.chunk.content.split('\n').take(3) {
            println!("   > {}", &line[..line.len().min(80)]);
        }
        println!();
        let _ = abs_path;
    }
}
