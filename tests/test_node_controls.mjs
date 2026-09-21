import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { webcrypto } from 'node:crypto';

// A small DOM substitute for the workflow compatibility contract. Browser UI
// and actual HTTP/image rendering are covered separately with the local app.
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.value = ''; this.textContent = ''; }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute() {}
    addEventListener() {}
}
globalThis.document = { createElement: tag => new Element(tag) };
if (!globalThis.crypto) globalThis.crypto = webcrypto;
const code = await readFile(new URL('../js/node_controls.js', import.meta.url), 'utf8');
const { installNodeControls } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const graph = { _nodes: [], setDirtyCanvas() {}, getNodeById(id) { return this._nodes.find(n => n.id === id); } };
class Node {
    constructor(id) {
        this.id = id; this.type = 'GPTImage2'; this.graph = graph;
        this.size = [400, 400]; this.properties = {};
        this.widgets = [{ name: 'model', value: 'gpt-image-2' }, { name: 'prompt', value: 'original' }];
        graph._nodes.push(this);
    }
    configure(saved) {
        assert.equal(this.widgets.length, 2, 'old plugins must see only original widgets during configure');
        this.properties = structuredClone(saved.properties);
        saved.widgets_values.forEach((v, i) => { this.widgets[i].value = v; });
    }
    serialize() { return { properties: structuredClone(this.properties), widgets_values: this.widgets.map(w => w.value) }; }
    addDOMWidget(name, type, root, options) {
        const widget = { name, type, root, options }; this.widgets.push(widget); return widget;
    }
    computeSize() { return [400, 750]; }
    setSize(size) { this.size = size; }
}
let jobs = [];
const submissions = [];
const controls = installNodeControls({ graph }, { apiURL: path => path }, {
    getJobs: () => jobs, submit: (...args) => submissions.push(args), cancel: async () => {}, showHistory() {},
});
controls.wrap(Node);
const first = new Node(1);
controls.attach(first);
const identity = first.properties.apiImmediateNodeId;
const saved = first.serialize();
assert.deepEqual(saved.widgets_values, ['gpt-image-2', 'original']);
assert.equal(first.widgets.length, 3, 'control restored after serialization');
saved.widgets_values[1] = 'edited prompt';
first.configure(saved);
assert.equal(first.widgets[1].value, 'edited prompt');
assert.equal(first.widgets.length, 3);
const clone = new Node(2);
clone.configure(saved);
controls.attach(clone);
assert.notEqual(clone.properties.apiImmediateNodeId, identity, 'clones have separate histories');
controls.attach(first);
assert.equal(first.widgets.length, 3, 'controls must not duplicate');

jobs = [
    { id: 'first-job', client_ref: identity, status: 'running', created: 1, items: [] },
    { id: 'second-job', client_ref: identity, status: 'completed', created: 2, items: [
        { index: 0, status: 'completed', images: [{ filename: 'test.png', type: 'output', subfolder: 'api_immediate' }] }] },
];
controls.updateAll(jobs);
assert.match(first._apiImmediateControls.summary.textContent, /1 批/);
assert.equal(first._apiImmediateControls.root.children.length, 3, 'only controls, summary and brief message remain');
assert.equal(first._apiImmediateControls.results, undefined);
assert.equal(first._apiImmediateControls.select, undefined);
first._apiImmediateControls.root.children[0].children[0].onclick();
first._apiImmediateControls.root.children[0].children[0].onclick();
assert.equal(submissions.length, 2, 'start stays enabled while running');
console.log('PASS: widget values preserved, configure/serialize restored, clone history isolated, controls unique, compact controls without inline logs, repeated start.');
