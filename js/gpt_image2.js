// ComfyUI-GPT-Image-2
// 根据 model 下拉框动态切换 size / quality 的可选项，并控制 background 是否显示。
//
// 数据来源是后端 GET /gpt-image2/models（model_config.py），避免 JS 里再抄一份导致漂移。
// 若接口不可用，这里什么都不做 —— 下拉框保留全部选项，Python 侧仍会做参数收敛。

import { app } from "../../scripts/app.js";

const NODE_TYPE = "GPTImage2";
const REGISTRY_URL = "/gpt-image2/models";

let registryPromise = null;

function loadRegistry() {
  if (!registryPromise) {
    registryPromise = fetch(REGISTRY_URL)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && data.models) return data;
        throw new Error("bad registry payload");
      })
      .catch((err) => {
        console.warn("[GPT Image 2] 模型能力表加载失败，跳过动态下拉:", err);
        return null;
      });
  }
  return registryPromise;
}

function findWidget(node, name) {
  return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

// ---------------------------------------------------------------------------
// 旧工作流迁移
//
// 加新 widget 会打乱 widgets_values 的**位置**映射：ComfyUI 前端的
// migrateWidgetsValues 只在「存档长度 == 当前 widget 数」时才按名字迁移，
// 长度对不上就退化成按下标赋值，于是每个参数都往后错一位（seed 会收到
// 'randomize' 或 undefined，直接报 "Failed to convert ... to a INT value"）。
//
// 解决：在 configure 结束时按「旧版本的名字序列」把存档值重排回去。
// 存档长度是唯一的版本指纹，8 = 加 num_images 之前，9 = 加 num_images 之后。
// ---------------------------------------------------------------------------
const LEGACY_LAYOUTS = {
  8: ["api_key", "prompt", "size", "quality", "output_format", "output_compression",
      "seed", "control_after_generate"],
  9: ["api_key", "prompt", "size", "quality", "output_format", "output_compression",
      "seed", "control_after_generate", "num_images"],
};

function migrateLegacyWidgets(node, savedObj, registry, defaults) {
  const values = savedObj && savedObj.widgets_values;
  if (!Array.isArray(values)) return false;

  const widgets = node.widgets || [];
  if (!widgets.length) return false;

  // 前端已经按名字迁移过了（长度刚好对上），不用管
  if (values.length === widgets.length) return false;

  const layout = LEGACY_LAYOUTS[values.length];
  if (!layout) return false;

  // 双重校验：新布局里 index 1 是 model。若它已经是个合法模型名，说明不是旧存档
  if (registry && registry.models && values[1] && values[1] in registry.models) return false;

  const byName = {};
  values.forEach((v, i) => {
    if (layout[i]) byName[layout[i]] = v;
  });

  let count = 0;
  for (const w of widgets) {
    if (Object.prototype.hasOwnProperty.call(byName, w.name)) {
      w.value = byName[w.name]; // 旧存档里有 -> 按名字还原
    } else if (Object.prototype.hasOwnProperty.call(defaults, w.name)) {
      w.value = defaults[w.name]; // 新增的 widget（model / background）-> 复位成默认值
    } else {
      continue;
    }
    count++;
  }
  if (count) {
    console.log(`[GPT Image 2] 已把旧工作流的 ${count} 个参数迁移到新布局（model 沿用默认 ${defaults.model || "gpt-image-2"}）`);
  }
  return count > 0;
}

function applyOptions(widget, values, preferred) {
  if (!widget || !Array.isArray(values) || values.length === 0) return false;

  const current = widget.value;
  const same =
    Array.isArray(widget.options?.values) &&
    widget.options.values.length === values.length &&
    widget.options.values.every((v, i) => v === values[i]);

  // litegraph 的 combo 用 options.values 渲染
  if (widget.options) widget.options.values = values.slice();
  else widget.options = { values: values.slice() };

  let changed = false;
  if (!values.includes(current)) {
    const next = preferred && values.includes(preferred) ? preferred : values[0];
    widget.value = next;
    changed = true;
  }

  if (!same || changed) {
    if (typeof widget.callback === "function" && changed) {
      try {
        widget.callback(widget.value);
      } catch (e) {
        /* noop */
      }
    }
    if (typeof widget.computeSize === "function") {
      try {
        widget.computeSize();
      } catch (e) {
        /* noop */
      }
    }
  }
  return changed;
}

function setVisible(node, widget, visible) {
  if (!widget) return;
  const changed = (widget.hidden === true) === visible;
  widget.hidden = !visible;
  if (typeof widget.computeSize === "function") {
    try {
      widget.computeSize();
    } catch (e) {
      /* noop */
    }
  }
  if (changed && node.size) {
    const [, h] = typeof node.computeSize === "function" ? node.computeSize() : [node.size[0], node.size[1]];
    node.setSize([node.size[0], Math.max(h, node.size[1] - 0)]);
  }
}

function applyModelConfig(node, registry, key, opts = {}) {
  const cfg = registry?.models?.[key] || registry?.models?.[registry?.default];
  if (!cfg) return;

  const sizeW = findWidget(node, "size");
  const qualityW = findWidget(node, "quality");
  const bgW = findWidget(node, "background");

  const sizePref = cfg.default_size || (cfg.sizes && cfg.sizes[0]);
  const qualityPref = cfg.default_quality || "medium";

  // 切换模型时才自动顶到该模型的默认值；加载旧工作流时保留用户存的值
  applyOptions(sizeW, cfg.sizes, opts.forceDefaults ? sizePref : undefined);
  applyOptions(qualityW, cfg.qualities, opts.forceDefaults ? qualityPref : undefined);

  setVisible(node, bgW, !!cfg.background);

  node._gptImage2 = {
    version: cfg.version,
    line: cfg.line,
    id: cfg.id,
    note: cfg.note,
  };
}

app.registerExtension({
  name: "ComfyUI.GPTImage2.ModelSwitch",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    const registry = await loadRegistry();

    // 从 object_info 里取每个 widget 的默认值，用于把新增 widget 复位
    const defaults = {};
    for (const [name, spec] of Object.entries(nodeData.inputs || {})) {
      const opts = Array.isArray(spec) ? spec[1] : null;
      if (opts && typeof opts === "object" && "default" in opts) defaults[name] = opts.default;
    }

    // 旧存档的位置迁移：在 configure 结束后按旧名字序列重排
    const proto = nodeType.prototype;
    const prevOnConfigure = proto.onConfigure;
    proto.onConfigure = function (o) {
      if (typeof prevOnConfigure === "function") prevOnConfigure.call(this, o);
      try {
        migrateLegacyWidgets(this, o, registry, defaults);
      } catch (e) {
        console.warn("[GPT Image 2] 旧工作流迁移失败:", e);
      }
    };
  },

  async nodeCreated(node) {
    if (node.type !== NODE_TYPE) return;

    const registry = await loadRegistry();
    if (!registry) return;

    const modelW = findWidget(node, "model");
    if (!modelW) return;

    // 首帧：沿用工作流里已存的值，不要强改
    applyModelConfig(node, registry, modelW.value, { forceDefaults: false });

    const prev = modelW.callback;
    modelW.callback = (value, ...rest) => {
      if (typeof prev === "function") prev(value, ...rest);
      try {
        applyModelConfig(node, registry, value, { forceDefaults: true });
        app.graph.setDirtyCanvas(true, true);
      } catch (e) {
        console.warn("[GPT Image 2] 切换模型参数失败:", e);
      }
    };
  },
});
