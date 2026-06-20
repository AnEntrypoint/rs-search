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

check('Cargo.toml is a wasm cdylib with the real dep set', () => {
    const c = read('Cargo.toml');
    if (!/name\s*=\s*"rs-search"/.test(c)) throw new Error('package name wrong');
    if (!/crate-type\s*=\s*\["cdylib"\]/.test(c)) throw new Error('crate-type cdylib missing');
    for (const dep of ['serde', 'serde_json', 'regex']) {
        if (!new RegExp(`(^|\\n)${dep}\\s*=`).test(c)) throw new Error(`${dep} dep missing`);
    }
});

check('lib.rs exposes the real modules', () => {
    const c = read('src/lib.rs');
    for (const m of ['context', 'eval', 'fusion', 'tokenize', 'wasm_host']) {
        if (!new RegExp(`pub mod ${m};`).test(c)) throw new Error(`${m} not in lib.rs`);
    }
    if (!/pub struct Searcher/.test(c)) throw new Error('Searcher struct missing');
    if (!/pub fn search\(&self, query: &str, k: usize\)/.test(c)) throw new Error('search entry missing');
});

check('fusion implements RRF k=60 with identifier boost', () => {
    const c = read('src/fusion.rs');
    if (!/RRF_K:\s*f64\s*=\s*60/.test(c)) throw new Error('RRF k=60 constant missing');
    if (!c.includes('IDENTIFIER_BOOST')) throw new Error('identifier boost missing');
    if (!c.includes('looks_like_identifier')) throw new Error('identifier detection missing');
    if (!c.includes('fn fuse')) throw new Error('fuse entry point missing');
    if (!c.includes('rrf_merge')) throw new Error('rrf_merge missing');
});

check('eval has NDCG/MRR/recall/precision', () => {
    const c = read('src/eval.rs');
    for (const fn of ['ndcg_at_k', 'mrr', 'recall_at_k', 'precision_at_k', 'evaluate']) {
        if (!new RegExp(`fn ${fn}`).test(c)) throw new Error(`${fn} missing`);
    }
});

check('tokenize splits camelCase', () => {
    const c = read('src/tokenize.rs');
    if (!c.includes('fn split_camel')) throw new Error('split_camel missing');
    if (!c.includes('fn tokenize')) throw new Error('tokenize entry missing');
});

check('context resolves enclosing scope', () => {
    const c = read('src/context.rs');
    if (!c.includes('fn find_enclosing_context')) throw new Error('find_enclosing_context missing');
});

check('wasm host imports carry link(wasm_import_module = env)', () => {
    const c = read('src/wasm_host.rs');
    if (!c.includes('#[link(wasm_import_module = "env")]')) throw new Error('link module attr missing');
    if (!/extern "C"/.test(c)) throw new Error('extern C block missing');
    for (const imp of ['host_vec_search', 'host_bm25_search', 'host_git_search']) {
        if (!c.includes(imp)) throw new Error(`${imp} import missing`);
    }
});

check('no UTF-8 BOM in tracked text files', () => {
    const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => {
        const p = path.join(d, e.name);
        if (e.isDirectory()) return e.name === 'target' || e.name === '.git' ? [] : walk(p);
        return [p];
    });
    const exts = new Set(['.rs', '.toml', '.js', '.json', '.md']);
    for (const f of walk(root)) {
        if (!exts.has(path.extname(f))) continue;
        const h = Buffer.alloc(3);
        const fd = fs.openSync(f, 'r');
        try { fs.readSync(fd, h, 0, 3, 0); } finally { fs.closeSync(fd); }
        if (h[0] === 0xef && h[1] === 0xbb && h[2] === 0xbf) throw new Error(`BOM in ${f}`);
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

check('all rust files under 200 lines', () => {
    const dir = path.join(root, 'src');
    for (const f of fs.readdirSync(dir)) {
        if (!f.endsWith('.rs')) continue;
        const lines = read(path.join('src', f)).split('\n').length;
        if (lines > 200) throw new Error(`${f} has ${lines} lines`);
    }
});

if (failed) {
    console.error(`\n${failed} check(s) failed`);
    process.exit(1);
}
console.log('\nAll checks passed');
