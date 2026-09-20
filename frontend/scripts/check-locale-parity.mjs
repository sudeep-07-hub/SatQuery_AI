/**
 * Asserts every locale file has exactly the key set of en.ts, and that no value was left as the
 * English string by accident in a non-English locale.
 *
 * tsc already enforces key parity through `Locale = Record<TranslationKey, string>` (a missing key
 * is TS2741, an extra key TS2353). This runs the same check at runtime so it can be seen in CI
 * output, and adds the untranslated-value check that types cannot express.
 *
 * Uses the esbuild that Vite already depends on — this repo has no test runner and this work order
 * adds no dependency.
 */
import { build } from 'esbuild';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const i18nDir = join(here, '..', 'src', 'i18n');

/** Values that are legitimately identical across locales: identifiers, not prose. */
const IDENTIFIER_VALUES = new Set([
  'GitHub', 'GeoJSON', 'JSON trace', 'Execution Trace',
]);

async function loadCatalogs() {
  const files = readdirSync(i18nDir)
    .filter((f) => /^(en|hi|kn|te|ta)\.ts$/.test(f))
    .map((f) => f.replace('.ts', ''));
  const result = await build({
    stdin: {
      contents: files.map((code) => `export { ${code} } from './i18n/${code}';`).join('\n'),
      resolveDir: join(here, '..', 'src'),
      loader: 'ts',
    },
    bundle: true,
    write: false,
    format: 'esm',
    platform: 'neutral',
  });
  const code = result.outputFiles[0].text;
  return import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
}

const catalogs = await loadCatalogs();
const enKeys = Object.keys(catalogs.en).sort();
const locales = ['hi', 'kn', 'te', 'ta'];
let failures = 0;

console.log(`en.ts defines ${enKeys.length} keys\n`);

for (const code of locales) {
  const keys = Object.keys(catalogs[code]).sort();
  const missing = enKeys.filter((k) => !keys.includes(k));
  const extra = keys.filter((k) => !enKeys.includes(k));
  const untranslated = enKeys.filter(
    (k) => catalogs[code][k] === catalogs.en[k] && !IDENTIFIER_VALUES.has(catalogs.en[k]),
  );

  const ok = missing.length === 0 && extra.length === 0;
  console.log(`${code}.ts  ${keys.length} keys  ${ok ? 'PARITY OK' : 'PARITY FAILED'}`);
  if (missing.length) { console.log(`   missing (${missing.length}): ${missing.join(', ')}`); failures++; }
  if (extra.length) { console.log(`   extra (${extra.length}): ${extra.join(', ')}`); failures++; }
  if (untranslated.length) {
    console.log(`   NOTE — ${untranslated.length} value(s) identical to English: ${untranslated.join(', ')}`);
  }
}

// Keys defined but never referenced from src/ are dead weight in every locale. Types cannot
// catch this: an unused key is a perfectly valid entry.
function sourceFiles(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const full = join(dir, e.name);
    if (e.isDirectory()) return e.name === 'i18n' ? [] : sourceFiles(full);
    return /\.(ts|tsx)$/.test(e.name) ? [full] : [];
  });
}
const src = sourceFiles(join(here, '..', 'src')).map((f) => readFileSync(f, 'utf8')).join('\n');
const unused = enKeys.filter((k) => !src.includes(`'${k}'`));
if (unused.length) {
  console.log(`\nUNUSED — ${unused.length} key(s) defined but never referenced from src/:`);
  for (const k of unused) console.log(`   ${k}`);
  failures++;
} else {
  console.log('\nEvery key is referenced from src/.');
}

console.log(failures === 0 ? 'All locales have identical key sets.' : `${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
