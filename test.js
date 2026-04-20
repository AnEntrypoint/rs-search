const fs = require('fs');
const path = require('path');

const root = __dirname;
let failed = 0;

const check = (name, fn) => {
    try {
        fn();
        console.log('PASS', name);
    } catch (e) {
        console.error('FAIL', name, '-', e.message);
        failed++;
    }
};

const read = (p) => fs.readFileSync(path.join(root, p), 'utf8');
const exists = (p) => fs.existsSync(path.join(root, p));

check('Cargo.toml has feature gates', () => {
    const c = read('Cargo.toml');
    if (!/\[features\]/.test(c)) throw new Error('[features] missing');
    if (!/default\s*=\s*\["vector", "perf"\]/.test(c)) throw new Error('default features wrong');
    if (!/vector\s*=\s*\["dep:candle-core"/.test(c)) throw new Error('vector feature missing');
    if (!/perf\s*=\s*\["dep:mimalloc"\]/.test(c)) throw new Error('perf feature missing');
});

check('Cargo.toml declares new deps', () => {
    const c = read('Cargo.toml');
    for (const dep of ['simsimd', 'blake3', 'ignore', 'arc-swap', 'mimalloc']) {
        if (!new RegExp(`${dep}\\s*=`).test(c)) throw new Error(`${dep} missing`);
    }
});

check('pdf-extract dep preserved', () => {
    if (!/pdf-extract\s*=\s*"0\.9"/.test(read('Cargo.toml'))) throw new Error('pdf-extract missing');
});

check('scanner uses ignore::WalkBuilder', () => {
    const c = read('src/scanner.rs');
    if (!c.includes('WalkBuilder')) throw new Error('WalkBuilder not used');
    if (!c.includes('git_ignore')) throw new Error('git_ignore not enabled');
    if (!c.includes('.codesearchignore')) throw new Error('custom ignore filename missing');
});

check('scanner still dispatches PDFs', () => {
    const c = read('src/scanner.rs');
    if (!c.includes('is_pdf')) throw new Error('is_pdf dispatch lost');
    if (!c.includes('pdf_chunks')) throw new Error('pdf_chunks call lost');
});

check('fusion module implements RRF k=60', () => {
    if (!exists('src/fusion.rs')) throw new Error('fusion.rs missing');
    const c = read('src/fusion.rs');
    if (!/RRF_K:\s*f64\s*=\s*60/.test(c)) throw new Error('RRF k=60 constant missing');
    if (!c.includes('looks_like_identifier')) throw new Error('identifier detection missing');
    if (!c.includes('IDENTIFIER_BOOST')) throw new Error('identifier boost missing');
    if (!c.includes('fn fuse')) throw new Error('fuse entry point missing');
});

check('embed_cache uses BLAKE3', () => {
    if (!exists('src/embed_cache.rs')) throw new Error('embed_cache.rs missing');
    const c = read('src/embed_cache.rs');
    if (!c.includes('blake3::Hasher')) throw new Error('BLAKE3 not used');
    if (!c.includes('emb-cache')) throw new Error('cache dir naming missing');
});

check('embed uses simsimd + MRL + prefixes + cache', () => {
    const c = read('src/embed.rs');
    if (!c.includes('SpatialSimilarity')) throw new Error('simsimd not used');
    if (!c.includes('truncate_mrl')) throw new Error('MRL truncation missing');
    if (!c.includes('query_prefix')) throw new Error('configurable query prefix missing');
    if (!c.includes('RS_SEARCH_DIM')) throw new Error('dim env var missing');
    if (!c.includes('EmbedCache')) throw new Error('embed cache integration missing');
});

check('mcp has panic boundary', () => {
    const c = read('src/mcp.rs');
    if (!c.includes('catch_unwind')) throw new Error('catch_unwind missing');
    if (!c.includes('-32603')) throw new Error('internal-error code missing');
});

check('explain module exists', () => {
    if (!exists('src/explain.rs')) throw new Error('explain.rs missing');
    const c = read('src/explain.rs');
    if (!c.includes('TokenBreakdown')) throw new Error('TokenBreakdown missing');
    if (!c.includes('looks_like_identifier')) throw new Error('identifier flag missing');
});

check('eval module has NDCG/MRR/recall', () => {
    if (!exists('src/eval.rs')) throw new Error('eval.rs missing');
    const c = read('src/eval.rs');
    for (const fn of ['ndcg_at_k', 'mrr', 'recall_at_k', 'precision_at_k']) {
        if (!new RegExp(`fn ${fn}`).test(c)) throw new Error(`${fn} missing`);
    }
});

check('main.rs uses mimalloc under feature', () => {
    const c = read('src/main.rs');
    if (!c.includes('#[cfg(feature = "perf")]')) throw new Error('perf cfg missing');
    if (!c.includes('mimalloc::MiMalloc')) throw new Error('MiMalloc allocator missing');
});

check('main.rs has serve/explain/search subcommands', () => {
    const c = read('src/main.rs');
    if (!c.includes('Command::Serve')) throw new Error('Serve subcommand missing');
    if (!c.includes('Command::Explain')) throw new Error('Explain subcommand missing');
});

check('bm25 delegates to ignore::is_code_file', () => {
    const c = read('src/bm25.rs');
    if (/fn is_code_file\s*\([^)]*\)\s*->\s*bool\s*\{[\s\S]*code_exts/.test(c))
        throw new Error('bm25 still has duplicated is_code_file table');
});

check('lib.rs exposes all new modules', () => {
    const c = read('src/lib.rs');
    for (const m of ['embed_cache', 'eval', 'explain', 'fusion', 'tokenize']) {
        if (!new RegExp(`pub mod ${m};`).test(c)) throw new Error(`${m} not in lib.rs`);
    }
});

check('release-plz and cargo-dist configured', () => {
    if (!exists('release-plz.toml')) throw new Error('release-plz.toml missing');
    const c = read('Cargo.toml');
    if (!/\[workspace\.metadata\.dist\]/.test(c)) throw new Error('cargo-dist config missing');
});

check('all rust files under 200 lines', () => {
    const dir = path.join(root, 'src');
    for (const f of fs.readdirSync(dir)) {
        if (!f.endsWith('.rs')) continue;
        const lines = read(path.join('src', f)).split('\n').length;
        if (lines > 200) throw new Error(`${f} has ${lines} lines`);
    }
});

check('no // or /* comments in rust source', () => {
    const dir = path.join(root, 'src');
    for (const f of fs.readdirSync(dir)) {
        if (!f.endsWith('.rs')) continue;
        for (const line of read(path.join('src', f)).split('\n')) {
            const t = line.trim();
            if (t.startsWith('//') || t.startsWith('/*')) throw new Error(`${f} has comment: ${t}`);
        }
    }
});

if (failed) {
    console.error(`\n${failed} check(s) failed`);
    process.exit(1);
}
console.log('\nAll checks passed');
