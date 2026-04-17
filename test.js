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

check('Cargo.toml declares pdf-extract', () => {
    const c = fs.readFileSync(path.join(root, 'Cargo.toml'), 'utf8');
    if (!/pdf-extract\s*=\s*"0\.9"/.test(c)) throw new Error('pdf-extract dep missing');
});

check('src/pdf.rs exists with extract_pages and pdf_chunks', () => {
    const p = path.join(root, 'src', 'pdf.rs');
    if (!fs.existsSync(p)) throw new Error('pdf.rs missing');
    const c = fs.readFileSync(p, 'utf8');
    if (!c.includes('pub fn pdf_chunks')) throw new Error('pdf_chunks missing');
    if (!c.includes('pub fn extract_pages')) throw new Error('extract_pages missing');
    if (!c.includes('extract_text_from_mem')) throw new Error('pdf-extract call missing');
    if (!c.includes('pdf-cache')) throw new Error('cache dir reference missing');
});

check('src/pdf.rs splits on form feed', () => {
    const c = fs.readFileSync(path.join(root, 'src', 'pdf.rs'), 'utf8');
    if (!c.includes("'\\u{000C}'")) throw new Error('form-feed split missing');
});

check('scanner dispatches PDFs', () => {
    const c = fs.readFileSync(path.join(root, 'src', 'scanner.rs'), 'utf8');
    if (!c.includes('is_pdf')) throw new Error('scanner missing is_pdf dispatch');
    if (!c.includes('pdf_chunks')) throw new Error('scanner missing pdf_chunks call');
});

check('lib.rs exposes pdf module', () => {
    const c = fs.readFileSync(path.join(root, 'src', 'lib.rs'), 'utf8');
    if (!c.includes('pub mod pdf;')) throw new Error('pdf module not declared');
});

check('ignore.rs treats .pdf as code, not binary', () => {
    const c = fs.readFileSync(path.join(root, 'src', 'ignore.rs'), 'utf8');
    const codeBlock = c.match(/CODE_EXTENSIONS[\s\S]*?\];/)[0];
    const binBlock = c.match(/BINARY_EXTENSIONS[\s\S]*?\];/)[0];
    if (!codeBlock.includes('".pdf"')) throw new Error('.pdf missing from CODE_EXTENSIONS');
    if (binBlock.includes('".pdf"')) throw new Error('.pdf still in BINARY_EXTENSIONS');
});

check('Chunk line_start=line_end=page convention', () => {
    const c = fs.readFileSync(path.join(root, 'src', 'pdf.rs'), 'utf8');
    if (!/line_start:\s*page_num/.test(c)) throw new Error('line_start not set to page_num');
    if (!/line_end:\s*page_num/.test(c)) throw new Error('line_end not set to page_num');
});

check('files under 200 lines', () => {
    for (const f of ['pdf.rs', 'scanner.rs', 'lib.rs']) {
        const lines = fs.readFileSync(path.join(root, 'src', f), 'utf8').split('\n').length;
        if (lines > 200) throw new Error(`${f} has ${lines} lines`);
    }
});

check('no comments in new rust files', () => {
    for (const f of ['pdf.rs', 'scanner.rs']) {
        const c = fs.readFileSync(path.join(root, 'src', f), 'utf8');
        for (const line of c.split('\n')) {
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
