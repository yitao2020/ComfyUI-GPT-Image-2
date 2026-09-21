// Transient controls are excluded from both workflow widget values and API inputs.
const CONTROL_NAME = '__api_immediate_controls';
const ACTIVE = new Set(['preparing', 'running']);


export function installNodeControls(app, api, { submit, showHistory, getJobs }) {
    function el(tag, text, className) {
        const item = document.createElement(tag);
        if (text) item.textContent = text;
        if (className) item.className = className;
        return item;
    }
    function btn(text, action, className) {
        const item = el('button', text, className);
        item.type = 'button';
        item.onclick = action;
        return item;
    }
    function matching(node, jobs) {
        const ref = node.properties?.apiImmediateNodeId;
        return jobs.filter(job => ref && job.client_ref === ref).reverse();
    }
    function update(node, jobs = getJobs(), force = false) {
        const ui = node._apiImmediateControls;
        if (!ui) return;
        const own = matching(node, jobs);
        const signature = JSON.stringify(own);
        if (!force && ui.signature === signature) return;
        ui.signature = signature;
        const active = own.filter(job => ACTIVE.has(job.status));
        ui.summary.textContent = active.length ? `${active.length} 批正在运行 · 可继续提交新任务` : '可以生成 · 每次点击启动独立任务';

    }
    function attach(node) {
        if (node._apiImmediateControls || !node.graph) return;
        node.properties ||= {};
        const shrinkLegacyPanel = !!node.properties.apiImmediateNodeId && !node.properties.apiImmediateCompact;
        // Cloned nodes must not inherit the original node's task history.
        const duplicate = (node.graph._nodes || []).some(other => other !== node &&
            other.properties?.apiImmediateNodeId === node.properties.apiImmediateNodeId);
        if (!node.properties.apiImmediateNodeId || duplicate) node.properties.apiImmediateNodeId = crypto.randomUUID();
        const root = el('div', '', 'api-node-controls');
        root.setAttribute('aria-label', `${node.comfyClass || node.type} 节点并发控制`);
        const top = el('div', '', 'api-node-toolbar');
        const start = btn('▶ 立即生成', () => {
            submit([node.id], { inline: true });
        }, 'api-node-start');
        start.title = '独立启动本次任务，不等待正在运行的批次';
        top.append(start, btn('全部记录', showHistory));
        const summary = el('div', '', 'api-node-summary');
        summary.setAttribute('aria-live', 'polite');
        const message = el('div', '', 'api-node-message');
        root.append(top, summary, message);
        for (const event of ['pointerdown', 'mousedown', 'dblclick', 'keydown', 'wheel']) {
            root.addEventListener(event, e => e.stopPropagation());
        }
        const height = 110;
        const originalHeight = Math.max(0, node.size[1] - (shrinkLegacyPanel ? 240 : 0));
        node.properties.apiImmediateCompact = true;
        const widget = node.addDOMWidget(CONTROL_NAME, 'api_immediate', root, {
            serialize: false, getValue: () => undefined, setValue: () => {},
            getMinHeight: () => height, getMaxHeight: () => height,
        });
        widget.serializeValue = () => undefined;
        node._apiImmediateControls = { widget, root, summary, message, signature: null };
        const computed = node.computeSize();
        node.setSize([Math.max(node.size[0], 390), Math.max(originalHeight, computed[1])]);
        update(node, getJobs(), true);
        app.graph?.setDirtyCanvas(true, true);
    }
    function wrap(nodeType) {
        const proto = nodeType.prototype;
        // Hiding the transient widget during configure/serialize preserves existing
        // plugins' positional widget migrations and clone/save/reload behavior.
        for (const method of ['configure', 'serialize']) {
            const original = proto[method];
            proto[method] = function (...args) {
                const widgets = this.widgets;
                this.widgets = widgets?.filter(widget => widget.name !== CONTROL_NAME);
                try { return original.apply(this, args); }
                finally { this.widgets = widgets; }
            };
        }
        const onAdded = proto.onAdded;
        proto.onAdded = function (...args) {
            const result = onAdded?.apply(this, args);
            setTimeout(() => attach(this), 0);
            return result;
        };
    }
    return { wrap, attach, update,
        message(ids, text) {
            for (const id of ids) {
                const ui = app.graph?.getNodeById(id)?._apiImmediateControls;
                if (ui) ui.message.textContent = text;
            }
        },
        updateAll(jobs) {
            for (const node of app.graph?._nodes || []) update(node, jobs);
        },
    };
}

export const NODE_STYLES = `
.api-node-controls {height:100%;width:100%;box-sizing:border-box;display:flex;flex-direction:column;gap:7px;background:#152033;border:1px solid #40516a;border-radius:9px;padding:10px;color:#e9f0fa;font:12px/1.45 system-ui,sans-serif;overflow:hidden;}
.api-node-toolbar {display:flex;gap:8px;align-items:center;flex-shrink:0;}
.api-node-controls button {padding:7px 10px;border:1px solid #55749e;border-radius:6px;background:#2b3b54;color:#eef6ff;cursor:pointer;white-space:nowrap;font:inherit;}
.api-node-controls button.api-node-start {background:#226fd1;font-weight:600;flex:1;}
.api-node-controls button:disabled {opacity:.4;cursor:default;}
.api-node-summary {color:#b8d7ff;flex-shrink:0;}
.api-node-message:empty {display:none;}
.api-node-message {color:#f6d48d;max-height:45px;overflow:auto;flex-shrink:0;overflow-wrap:anywhere;}
`;
