import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';

const code = (await readFile(new URL('../js/immediate.js', import.meta.url), 'utf8'))
    .split('\n').filter(line => !line.startsWith('import ')).join('\n');
const run = new Function('app', 'api', 'installNodeControls', 'NODE_STYLES',
    'installResultNodes', 'LiteGraph', 'sessionStorage', 'crypto', 'globalThis', code);
const registrations = [];
const app = { registerExtension: extension => registrations.push(extension) };
const storage = new Map();
const sessionStorage = { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) };
const page = {};
const execute = context => run(app, {}, () => ({}), '', () => ({}), {}, sessionStorage, webcrypto, context);
execute(page); // First plugin bundle.
execute(page); // Other plugin imports its independent copy of these files.
assert.equal(registrations.length, 1, 'two plugin bundles must register only one extension');
assert.equal(typeof registrations[0].registerCustomNodes, 'function');
assert.equal(typeof registrations[0].beforeRegisterNodeDef, 'function');
execute({}); // Installing just the other bundle on a fresh page also works.
assert.equal(registrations.length, 2);
console.log('PASS: single-plugin registration and duplicate frontend bundle suppression.');
