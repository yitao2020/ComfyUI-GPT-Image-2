import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
const source = await readFile(new URL('../js/result_nodes.js', import.meta.url), 'utf8');
const { installResultNodes, RESULT_TYPE } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
globalThis.Image = class { set src(value) { this.address = value; } };
class Base {
    constructor() { this.widgets = []; this.flags = {}; }
    addInput() {}
    addWidget(type, name, value, callback, options = {}) { const w = {type, name, value, callback, options}; this.widgets.push(w); return w; }
    setDirtyCanvas() {}
}
const definitions = {};
const LiteGraph = { LGraphNode: Base,
    registerNodeType(name, cls) { definitions[name] = cls; },
    createNode(name) { const node = new definitions[name](); node.type = name; return node; } };
const graph = { _nodes: [], add(node) { this._nodes.push(node); node.graph = this; }, setDirtyCanvas() {}, change() {} };
const generator = { pos: [0, 0], size: [400, 600], properties: { apiImmediateNodeId: 'source' }, outputs: [{ type: 'IMAGE' }], connect(slot, target, port) { target.connection = [slot, port]; } };
graph.add(generator);
const result = installResultNodes({ graph, canvas: { centerOnNode() {} } }, { apiURL: path => path }, LiteGraph);
result.register();
const first = result.create(generator, graph);
const second = result.create(generator, graph);
assert.notEqual(first, second);
assert.equal(first.type, RESULT_TYPE);
assert.equal(first.isVirtualNode, true, 'output cards never enter the native workflow executor');
assert.ok(second.pos[1] >= first.pos[1] + first.size[1], 'new batches do not overlap earlier result nodes');
assert.deepEqual(first.connection, [0, 0]);
result.bind(first, 'job-A'); result.bind(second, 'job-B');
const image = { filename: 'a.png', subfolder: 'api_immediate', type: 'output' };
result.updateAll([{ id: 'job-B', status: 'completed', items: [{ status: 'completed', images: [image] }] },
    { id: 'job-A', status: 'running', items: [] }]);
assert.equal(first.properties.snapshot.status, 'running');
assert.equal(second.files()[0].filename, 'a.png');
result.updateAll([{ id: 'job-A', status: 'completed', items: [{ status: 'completed', images: [{...image, filename: 'b.png'}] }] }]);
assert.equal(first.files()[0].filename, 'b.png');
assert.equal(second.files()[0].filename, 'a.png', 'another job never overwrites an older result');
const restored = LiteGraph.createNode(RESULT_TYPE);
restored.properties = structuredClone(second.properties);
restored.onConfigure();
assert.equal(restored.files()[0].filename, 'a.png', 'saved workflow restores previews without server history');
result.fail(first, 'test failure');
assert.equal(first.properties.snapshot.status, 'failed');
assert.equal(second.properties.snapshot.status, 'completed');
console.log('PASS: automatic output nodes, separate placement/links, virtual execution, out-of-order results, saved preview restoration, failure isolation.');
