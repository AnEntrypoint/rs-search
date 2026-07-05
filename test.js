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

check('Cargo.toml is a dependency-free rlib consumed by rs-plugkit', () => {
    const c = read('Cargo.toml');
    if (!/name\s*=\s*"rs-search"/.test(c)) throw new Error('package name wrong');
    if (!/crate-type\s*=\s*\["rlib"\]/.test(c)) throw new Error('crate-type rlib missing');
    if (/\[dependencies\]/.test(c)) throw new Error('crate must stay dependency-free');
});

check('lib.rs exposes exactly the consumed modules', () => {
    const c = read('src/lib.rs');
    for (const m of ['fusion', 'tokenize']) {
        if (!new RegExp(`pub mod ${m};`).test(c)) throw new Error(`${m} not in lib.rs`);
    }
    for (const dead of ['context', 'eval', 'wasm_host', 'Searcher', 'SearchHit']) {
        if (c.includes(dead)) throw new Error(`consumer-free surface ${dead} must stay deleted`);
    }
});

check('fusion implements RRF k=60 with identifier boost', () => {
    const c = read('src/fusion.rs');
    if (!/RRF_K:\s*f64\s*=\s*60/.test(c)) throw new Error('RRF k=60 constant missing');
    if (!c.includes('IDENTIFIER_BOOST')) throw new Error('identifier boost missing');
    if (!c.includes('looks_like_identifier')) throw new Error('identifier detection missing');
    if (!c.includes('fn fuse_n')) throw new Error('fuse_n entry point missing');
    if (!c.includes('fn rrf_merge_n_weighted')) throw new Error('rrf_merge_n_weighted missing');
    if (!c.includes('fn rrf_merge_n')) throw new Error('rrf_merge_n missing');
});

check('tokenize splits camelCase and strips punctuation from camel tokens', () => {
    const c = read('src/tokenize.rs');
    if (!c.includes('fn split_camel')) throw new Error('split_camel missing');
    if (!c.includes('fn tokenize')) throw new Error('tokenize entry missing');
    if (!c.includes('fn add_word_tokens')) throw new Error('add_word_tokens missing');
    if (!/for p in t\.split\(\|c: char\| !c\.is_alphanumeric\(\)\)/.test(c)) throw new Error('camel tokens must be split on non-alphanumerics');
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

check('no // or /* comments and no cfg(test) in rust source', () => {
    const dir = path.join(root, 'src');
    for (const f of fs.readdirSync(dir)) {
        if (!f.endsWith('.rs')) continue;
        const c = read(path.join('src', f));
        if (c.includes('#[cfg(test)]')) throw new Error(`${f} has synthetic test module`);
        for (const line of c.split('\n')) {
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
