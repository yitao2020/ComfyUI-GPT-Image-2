export const RESULT_TYPE = 'APIImmediateSaveImage';
const LABELS = { submitting: '正在提交', preparing: '准备输入', running: '生成中',
    completed: '已保存', partial: '部分成功', failed: '失败', cancelled: '已停止接收' };

export function installResultNodes(app, api, LiteGraph) {
    function url(image) { return api.apiURL(`/view?${new URLSearchParams(image)}`); }
    class ResultNode extends LiteGraph.LGraphNode {
        constructor() {
            super();
            this.title = '保存图像 · API 结果';
            this.isVirtualNode = true;
            this.serialize_widgets = false;
            this.properties = { snapshot: { status: 'submitting', items: [] } };
            this.addInput('images', 'IMAGE');
            this._previews = new Map();
            this._selected = 0;
            this._selector = this.addWidget('combo', '查看图片', '第 1 张', value => {
                this._selected = Math.max(0, Number(value.match(/\d+/)?.[0] || 1) - 1);
            }, { values: ['第 1 张'] });
            this.addWidget('button', '打开所选原图', null, () => {
                const image = this.files()[this._selected];
                if (image) window.open(url(image), '_blank', 'noopener');
            });
            this.size = [460, 460];
            this.color = '#25405d';
            this.bgcolor = '#172334';
        }
        files() { return (this.properties.snapshot?.items || []).flatMap(item => item.images || []); }
        setSnapshot(snapshot) {
            const signature = JSON.stringify(snapshot);
            if (signature === this._signature) return;
            this._signature = signature;
            // Persist only output file descriptors and status, never credentials or the input workflow.
            this.properties.snapshot = structuredClone(snapshot);
            this.title = `保存图像 · ${LABELS[snapshot.status] || snapshot.status} · ${(snapshot.id || '').slice(0, 6)}`;
            const files = this.files();
            this._selector.options.values = files.length ? files.map((_, i) => `第 ${i + 1} 张`) : ['第 1 张'];
            if (this._selected >= files.length) this._selected = 0;
            this._selector.value = `第 ${this._selected + 1} 张`;
            for (const file of files) {
                const address = url(file);
                if (this._previews.has(address)) continue;
                const image = new Image();
                this._previews.set(address, image);
                image.onload = () => this.setDirtyCanvas(true, true);
                image.onerror = () => this.setDirtyCanvas(true, true);
                image.src = address;
            }
            this.setDirtyCanvas(true, true);
            if (['completed', 'partial', 'failed', 'cancelled'].includes(snapshot.status)) this.graph?.change?.();
        }
        onConfigure() {
            this.isVirtualNode = true;
            this._signature = null;
            this.setSnapshot(this.properties.snapshot || { status: 'submitting', items: [] });
        }
        onDrawForeground(ctx) {
            if (this.flags.collapsed) return;
            const job = this.properties.snapshot || {};
            const items = job.items || [];
            const tiles = items.flatMap(item => item.images?.length
                ? item.images.map(image => ({ ...item, image })) : [item]);
            if (!tiles.length) tiles.push({ status: job.status, error: job.error });
            const width = this.size[0], height = this.size[1];
            ctx.save();
            ctx.beginPath(); ctx.rect(0, 76, width, Math.max(0, height - 76)); ctx.clip();
            ctx.font = '12px sans-serif'; ctx.fillStyle = '#bfd3ed';
            ctx.fillText(job.error || `${LABELS[job.status] || '等待状态'} · 本节点仅展示这一批 · 返回图片后自动保存`, 12, 96, width - 24);
            const columns = tiles.length > 1 ? 2 : 1;
            const rows = Math.ceil(tiles.length / columns);
            const cellWidth = (width - 24 - (columns - 1) * 8) / columns;
            const cellHeight = (height - 126 - (rows - 1) * 8) / rows;
            tiles.forEach((tile, i) => {
                const x = 12 + (i % columns) * (cellWidth + 8);
                const y = 110 + Math.floor(i / columns) * (cellHeight + 8);
                ctx.fillStyle = '#23344c'; ctx.fillRect(x, y, cellWidth, cellHeight);
                ctx.fillStyle = tile.error ? '#ffb7b7' : '#d6e5fa';
                ctx.fillText(`第 ${i + 1} 张 · ${LABELS[tile.status] || '准备中'}`, x + 8, y + 18, cellWidth - 16);
                const image = tile.image && this._previews.get(url(tile.image));
                if (image?.complete && image.naturalWidth) {
                    const scale = Math.min((cellWidth - 16) / image.naturalWidth, Math.max(1, cellHeight - 36) / image.naturalHeight);
                    const w = image.naturalWidth * scale, h = image.naturalHeight * scale;
                    ctx.drawImage(image, x + (cellWidth - w) / 2, y + 28 + (cellHeight - 36 - h) / 2, w, h);
                } else {
                    const text = tile.error || (tile.image ? '正在载入预览（可打开原图）' : '等待本张生成结果…');
                    // Wrap enough detail to make API failures visible without spilling outside this tile.
                    const chars = Math.max(8, Math.floor((cellWidth - 16) / 8));
                    for (let offset = 0, line = 0; offset < text.length && 42 + line * 16 < cellHeight; offset += chars, line++) {
                        ctx.fillText(text.slice(offset, offset + chars), x + 8, y + 42 + line * 16, cellWidth - 16);
                    }
                }
            });
            ctx.restore();
        }
    }
    function register() {
        LiteGraph.registerNodeType(RESULT_TYPE, Object.assign(ResultNode, { title: '保存图像 · API 结果' }));
        ResultNode.category = 'API 独立并发';
    }
    function create(source, graph) {
        const node = LiteGraph.createNode(RESULT_TYPE);
        if (!node) throw new Error('结果节点尚未注册，请刷新 ComfyUI 页面');
        const x = source.pos[0] + source.size[0] + 90;
        let y = source.pos[1];
        // Find the next free space to the right; existing batch nodes stay in place.
        for (;;) {
            const collision = graph._nodes.find(other => x < other.pos[0] + other.size[0] + 20 &&
                x + node.size[0] + 20 > other.pos[0] && y < other.pos[1] + other.size[1] + 20 &&
                y + node.size[1] + 20 > other.pos[1]);
            if (!collision) break;
            y = collision.pos[1] + collision.size[1] + 40;
        }
        node.pos = [x, y];
        node.properties.sourceRef = source.properties?.apiImmediateNodeId || '';
        graph.add(node);
        const imageSlot = source.outputs?.findIndex(output => output.type === 'IMAGE') ?? -1;
        if (imageSlot >= 0) source.connect(imageSlot, node, 0);
        node.setSnapshot({ status: 'submitting', items: [] });
        graph.setDirtyCanvas(true, true);
        graph.change?.();
        return node;
    }
    return { register, create,
        bind(node, id) { node.properties.jobId = id; node.setSnapshot({ id, status: 'preparing', items: [] }); node.graph?.change?.(); },
        fail(node, error) { node.setSnapshot({ status: 'failed', items: [], error }); node.graph?.change?.(); },
        updateAll(jobs) {
            const byId = new Map(jobs.map(job => [job.id, job]));
            for (const node of app.graph?._nodes || []) {
                if (node.type !== RESULT_TYPE) continue;
                const job = byId.get(node.properties.jobId);
                if (job) node.setSnapshot(job);
            }
        },
    };
}
