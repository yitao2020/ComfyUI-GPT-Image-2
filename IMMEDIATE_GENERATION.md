# 内置独立并发

本插件已经内置“立即生成”功能，不需要安装 ComfyUI-API-Immediate。
Nano Banana Pro 和 GPT Image 2 可以单独安装；同时安装时只注册一套任务接口和界面。
没有修改 ComfyUI 主体、原生队列或执行器。

## 使用

1. 更新插件后重启 ComfyUI，再用 Ctrl+F5 刷新浏览器。
2. 点击生成节点底部的 **▶ 立即生成**。上一批没完成时也能继续提交，多个节点可以同时生成。
3. 每次提交自动在画布上新增并连接一个 **保存图像 · API结果** 节点。
   画布保持当前位置和缩放，不自动跳转；每批结果独立，不会覆盖之前的图片。
4. 图片返回后逐张显示，并自动保存在 `output/api_immediate`。
   结果节点可选择图片、打开原图；保存工作流后重新打开仍可预览已保存的文件。
5. 生成节点只显示按钮和简短状态，不展示内嵌日志、批次列表或图片卡片。
   页面没有右下角悬浮按钮。需要历史记录或停止接收任务时，点击节点上的 **全部记录**。

原生“运行 / Queue Prompt”保持原来的执行方式。单次多图并发也照常可用。
“立即生成”走独立任务接口，不进入原生队列，也不执行原有下游 SaveImage、ShowText 或后处理节点。
新增的结果节点仅展示该批已保存的图片，不会重复调用 API。

## 兼容与边界

- 支持上游 LoadImage、LoadImageMask、LoadImageOutput、ImageBatch、BatchImagesNode、
  ImageScale、ImageScaleBy、ImageInvert、ImageCrop、MultiAngleCameraNode。
- 其他上游、API 串接、列表映射、本地模型节点目前会明确报错；不会自动改走原生队列。
- 沿用原节点的 1–9 张数量、模型、接口和超时参数。提交时保存参数快照，参考图片在准备阶段读入。
- Key 从节点或本插件的 `api_key.txt` 读取；并发模块不另存 Key，也不把输入工作流写入输出 PNG。
- 不自动重试付费请求。服务商额度、并发限额和远端排队仍由服务商决定。
- “停止接收”不能撤回已发出的远端请求；远端仍可能完成并计费。
- 当前标签页刷新可以恢复任务记录；服务重启后内存记录消失，保存的图片及工作流结果节点仍保留。

## 从独立插件迁移

将旧的 `ComfyUI-API-Immediate` 文件夹移出 `custom_nodes` 后再重启。
旧结果节点类型、任务接口路径、浏览器会话标识和输出目录都沿用原来的名称，保存的工作流可继续使用。

## 开发与测试

两个仓库各自包含同一份 `api_immediate/` 后端和 `js/immediate.js`、`js/node_controls.js`、
`js/result_nodes.js` 前端。后端按 ComfyUI 服务实例去重，前端按 ABI 标识去重。
修改通用并发行为时请同步这几份文件，并验证单独安装、双插件同时安装。
此设计不需要第三个仓库、子模块或额外安装步骤。

在本插件目录执行（接口被模拟，不会产生生图费用）：

```bash
python -X utf8 -m unittest discover -s tests -v
node tests/test_node_controls.mjs
node tests/test_result_nodes.mjs
node tests/test_frontend_registration.mjs
```
