import { app } from '../../scripts/app.js';
import { api } from '../../scripts/api.js';
import { installNodeControls, NODE_STYLES } from './node_controls.js';
import { installResultNodes } from './result_nodes.js';

const TYPES = new Set(['NanoBananaPro', 'GPTImage2']);
const SESSION_KEY = 'comfy.apiImmediate.session';
let session = sessionStorage.getItem(SESSION_KEY);
if (!session) {
    session = Array.from(crypto.getRandomValues(new Uint8Array(16)), n => n.toString(16).padStart(2, '0')).join('');
    sessionStorage.setItem(SESSION_KEY, session);
}
let panel, jobsElement, messageElement, lastState = '', polling = false;
let currentJobs = [];
const results = installResultNodes(app, api, LiteGraph);
const controls = installNodeControls(app, api, { submit,
    cancel: async id => { await request(`/jobs/${id}/cancel`, { method: 'POST' }); await refresh(); },
    showHistory: () => { show(); refresh(); }, getJobs: () => currentJobs });

async function request(path, options = {}) {
    const response = await api.fetchApi(`/api-immediate${path}`, {
        ...options,
        headers: { 'Content-Type': 'application/json', 'X-API-Immediate-Session': session },
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
    return data;
}

function element(tag, text, className) {
    const result = document.createElement(tag);
    if (text) result.textContent = text;
    if (className) result.className = className;
    return result;
}

function button(text, callback) {
    const result = element('button', text);
    result.type = 'button';
    result.onclick = callback;
    return result;
}

function show() { panel.hidden = false; }

async function submit(ids, { inline = false } = {}) {
    if (!inline) show();
    const notify = text => { messageElement.textContent = text; controls.message(ids, text); };
    const outputNodes = new Map();
    try {
        if (!ids.length) throw new Error('请选择 NanoBananaPro 或 GPTImage2 节点，或点击“运行全部 API 节点”。');
        notify('正在保存本次参数并提交…');
        const references = new Map(ids.map(id => [id, app.graph.getNodeById(id)?.properties?.apiImmediateNodeId || '']));
        const graph = app.graph;
        for (const id of ids) {
            const source = graph.getNodeById(id);
            if (!source) throw new Error('生成节点已被删除');
            outputNodes.set(id, results.create(source, graph));
        }
        const { output } = await app.graphToPrompt();
        // Clone before the first network await. The snapshot is not changed by later canvas edits.
        const prompt = structuredClone(output);
        const responses = await Promise.allSettled(ids.map(async target => {
            try {
                const response = await request('/jobs', {
                    method: 'POST', body: JSON.stringify({ prompt, target: String(target), client_ref: references.get(target) }),
                });
                results.bind(outputNodes.get(target), response.id);
                return response;
            } catch (error) { results.fail(outputNodes.get(target), error.message); throw error; }
        }));
        const failures = responses.filter(r => r.status === 'rejected');
        const accepted = responses.length - failures.length;
        notify(`已启动 ${accepted} 个独立任务。` + failures.map(r => r.reason.message).join('；'));
        await refresh();
    } catch (error) {
        for (const node of outputNodes.values()) results.fail(node, error.message);
        notify(error.message);
    }
}

function activeNodes(selected = false) {
    const candidates = selected ? Object.values(app.canvas?.selected_nodes || {}) : app.graph?._nodes || [];
    return candidates.filter(n => TYPES.has(n.comfyClass || n.type) && n.mode !== 2 && n.mode !== 4)
        .map(n => n.id);
}

const labels = { preparing: '准备输入', running: '请求中', completed: '完成', partial: '部分成功', failed: '失败', cancelled: '已停止接收' };

function render(jobs) {
    currentJobs = jobs;
    controls.updateAll(jobs);
    results.updateAll(jobs);
    const state = JSON.stringify(jobs);
    if (state === lastState) return;
    lastState = state;
    jobsElement.replaceChildren();
    if (!jobs.length) jobsElement.append(element('p', '还没有独立任务。可右键 API 节点 → 立即生成。'));
    for (const job of [...jobs].reverse()) {
        const card = element('section', '', 'api-immediate-job');
        card.append(element('strong', `${job.node_type} #${job.node_id} · ${labels[job.status] || job.status}`));
        card.append(element('small', `${new Date(job.created * 1000).toLocaleTimeString()} · ${job.id.slice(0, 8)}`));
        if (job.error) card.append(element('p', job.error, 'api-immediate-error'));
        if (['preparing', 'running'].includes(job.status)) {
            card.append(button('停止接收本批结果', async () => {
                try {
                    await request(`/jobs/${job.id}/cancel`, { method: 'POST' });
                    messageElement.textContent = '已停止接收；已发出的远端请求可能仍会完成并计费。';
                    await refresh();
                } catch (error) { messageElement.textContent = error.message; }
            }));
        }
        const grid = element('div', '', 'api-immediate-grid');
        for (const item of job.items) {
            const tile = element('div');
            tile.append(element('small', `第 ${item.index + 1} 张 · ${labels[item.status] || item.status}`));
            if (item.error) tile.append(element('p', item.error, 'api-immediate-error'));
            for (const image of item.images) {
                const url = api.apiURL(`/view?${new URLSearchParams(image)}`);
                const link = element('a');
                link.href = url;
                link.target = '_blank';
                link.rel = 'noopener';
                const img = element('img');
                img.src = url;
                img.alt = `生成结果 ${item.index + 1}`;
                img.loading = 'lazy';
                link.append(img);
                tile.append(link);
            }
            grid.append(tile);
        }
        card.append(grid);
        jobsElement.append(card);
    }
}

async function refresh() {
    if (polling) return;
    polling = true;
    try { render((await request('/jobs')).jobs); }
    catch (error) { if (panel && !panel.hidden) messageElement.textContent = `任务状态读取失败：${error.message}`; }
    finally { polling = false; }
}

// Either plugin can provide this extension; installing both must not double-submit.
const registrationKey = Symbol.for('comfy.apiImmediate.frontend.v1');
if (!globalThis[registrationKey]) {
globalThis[registrationKey] = true;
app.registerExtension({
    name: 'API.Immediate.NoQueue',
    registerCustomNodes() { results.register(); },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!TYPES.has(nodeData.name)) return;
        controls.wrap(nodeType);
        const original = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            const result = original?.apply(this, arguments);
            options.unshift({ content: '立即生成（API 独立并发）', callback: () => submit([this.id]) });
            return result;
        };
    },
    loadedGraphNode(node) { if (TYPES.has(node.comfyClass || node.type)) setTimeout(() => controls.attach(node), 0); },
    setup() {
        const style = element('style');
        style.textContent = `
        #api-immediate-panel {font:13px/1.5 system-ui,sans-serif;color:#e9eef7;z-index:10000;}
        #api-immediate-panel {position:fixed;right:18px;top:72px;width:min(460px,calc(100vw - 36px));max-height:calc(100vh - 160px);overflow:auto;background:#111b2c;border:1px solid #41516a;border-radius:12px;box-shadow:0 12px 40px #0008;padding:16px;box-sizing:border-box;}
        #api-immediate-panel[hidden] {display:none!important;}
        #api-immediate-panel header {display:flex;align-items:center;justify-content:space-between;font-size:17px;}
        #api-immediate-panel button {background:#26497a;color:#fff;border:1px solid #5878a2;border-radius:6px;padding:6px 10px;margin:4px 6px 4px 0;cursor:pointer;}
        #api-immediate-panel p {white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0;color:#bac9df;}
        #api-immediate-panel small {display:block;color:#a7b8d0;margin:5px 0;}
        .api-immediate-job {border-top:1px solid #344159;margin-top:14px;padding-top:12px;}
        .api-immediate-grid {display:grid;grid-template-columns:1fr 1fr;gap:9px;}
        .api-immediate-grid>div {min-width:0;background:#1b283d;border-radius:8px;padding:8px;}
        .api-immediate-grid img {width:100%;border-radius:5px;display:block;}
        #api-immediate-panel .api-immediate-error {color:#ffb3b3;font-size:12px;}
        ` + NODE_STYLES;
        document.head.append(style);
        panel = element('aside');
        panel.id = 'api-immediate-panel';
        panel.hidden = true;
        const header = element('header');
        header.append(element('strong', 'API 独立并发'), button('收起', () => { panel.hidden = true; }));
        panel.append(header, element('p', '每次点击启动新任务并新建保存图像结果节点，不等待上一批。图片自动保存在 output/api_immediate。'));
        panel.append(button('运行选中 API 节点', () => submit(activeNodes(true))),
            button('运行全部 API 节点', () => submit(activeNodes(false))));
        messageElement = element('p');
        panel.append(messageElement);
        jobsElement = element('div');
        panel.append(jobsElement);
        document.body.append(panel);
        refresh();
        setInterval(refresh, 1500);
    },
});

}
