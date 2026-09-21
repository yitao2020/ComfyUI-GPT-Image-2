import inspect
from pathlib import Path

TARGETS = {'NanoBananaPro', 'GPTImage2'}
# Only CPU input preparation. Never dispatch arbitrary model nodes outside the core executor.
INPUT_NODES = {'LoadImage', 'LoadImageMask', 'LoadImageOutput', 'ImageBatch',
               'ImageScale', 'ImageScaleBy', 'ImageInvert', 'ImageCrop',
               'MultiAngleCameraNode', 'BatchImagesNode'}


def is_link(value):
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and type(value[1]) is int)


def validate(payload, registry):
    graph, target = payload.get('prompt'), str(payload.get('target', ''))
    if not isinstance(graph, dict) or target not in graph:
        raise ValueError('缺少目标节点或工作流')
    if graph[target].get('class_type') not in TARGETS:
        raise ValueError('只支持 NanoBananaPro 和 GPTImage2')
    visiting, visited = set(), set()

    def walk(node_id):
        if node_id in visiting:
            raise ValueError('输入连线存在循环')
        if node_id in visited:
            return
        node = graph.get(node_id)
        if not isinstance(node, dict) or not isinstance(node.get('inputs'), dict):
            raise ValueError(f'节点 {node_id} 数据无效')
        kind = node.get('class_type')
        if node_id != target and kind not in INPUT_NODES:
            raise ValueError(f'上游节点 {kind} 尚未适配立即生成；请改用 LoadImage 或直接填写提示词')
        if kind not in registry:
            raise ValueError(f'节点 {kind} 未加载，请检查 ComfyUI 启动日志')
        cls = registry[kind]
        schema = cls.INPUT_TYPES()
        if kind == 'BatchImagesNode':
            # V3 autogrow is flattened to images.image0, images.image1, ... in API prompts.
            import re
            if not node['inputs'] or any(not re.fullmatch(r'images\.image\d+', k) for k in node['inputs']):
                raise ValueError('BatchImagesNode 动态输入格式不匹配')
            schema = {'required': {k: ('IMAGE',) for k in node['inputs']}}
        declared = {**schema.get('required', {}), **schema.get('optional', {})}
        unknown = set(node['inputs']) - set(declared)
        if unknown:
            raise ValueError(f'节点 {kind} 参数与安装版本不匹配，请从菜单重新添加该节点')
        for name in schema.get('required', {}):
            if name not in node['inputs']:
                raise ValueError(f'节点 {kind} 缺少参数 {name}，请重新添加节点')
        if node_id == target:
            for name, value in node['inputs'].items():
                if is_link(value):
                    continue
                spec = declared[name]
                expected = spec[0]
                options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                invalid = (isinstance(expected, list) and value not in expected)
                if expected == 'STRING':
                    invalid |= not isinstance(value, str)
                if expected in ('IMAGE', 'MASK'):
                    invalid = True
                if expected == 'INT':
                    invalid |= type(value) is not int
                    if not invalid:
                        invalid |= value < options.get('min', value) or value > options.get('max', value)
                if invalid:
                    raise ValueError(f'节点 {kind} 的 {name} 参数无效；旧工作流请重新添加该节点')
        visiting.add(node_id)
        for value in node['inputs'].values():
            if is_link(value):
                walk(value[0])
                returns = getattr(registry[graph[value[0]]['class_type']], 'RETURN_TYPES', ())
                if value[1] < 0 or value[1] >= len(returns):
                    raise ValueError('上游输出端口无效')
        visiting.remove(node_id)
        visited.add(node_id)
    walk(target)
    cls = registry[graph[target]['class_type']]
    method = '_generate_one' if graph[target]['class_type'] == 'NanoBananaPro' else 'send_single_request'
    if not hasattr(cls, method):
        raise ValueError('当前 API 插件版本与立即生成适配器不匹配')
    # Keep only the target and its actual dependencies; no other nodes or their keys.
    client_ref = payload.get('client_ref', '')
    if not isinstance(client_ref, str) or len(client_ref) > 64:
        raise ValueError('节点任务标识无效')
    return {'target': target, 'prompt': {key: graph[key] for key in visited}, 'client_ref': client_ref}


def prepare(payload, registry):
    graph, target = payload['prompt'], payload['target']
    cache = {}

    def inputs(node_id):
        values = {}
        for name, value in graph[node_id]['inputs'].items():
            if is_link(value):
                source, port = value
                if source not in cache:
                    cls = registry[graph[source]['class_type']]
                    source_inputs = inputs(source)
                    if graph[source]['class_type'] == 'BatchImagesNode':
                        ordered = sorted(source_inputs, key=lambda name: int(name.rsplit('image', 1)[1]))
                        nested = {name.split('.', 1)[1]: source_inputs[name] for name in ordered}
                        cache[source] = cls.execute(nested).result
                    else:
                        obj = cls()
                        cache[source] = getattr(obj, obj.FUNCTION)(**source_inputs)
                value = cache[source][port]
            values[name] = value
        return values

    kind = graph[target]['class_type']
    cls = registry[kind]
    args = inputs(target)
    # Apply Python defaults for optional inputs omitted by graphToPrompt.
    signature = inspect.signature(cls.generate_image)
    for name, param in signature.parameters.items():
        if name != 'self' and param.default is not inspect.Parameter.empty:
            args.setdefault(name, param.default)
    obj = cls()
    key = args.get('api_key', '').strip()
    if not key:
        key_path = Path(obj.key_file)
        key = key_path.read_text(encoding='utf-8').strip() if key_path.exists() else ''
    if not key:
        raise ValueError('Missing API key')
    # Read credentials without calling legacy get_api_key(), which writes shared files.
    if kind == 'NanoBananaPro':
        count = int(args.get('image_count', 1))
        def request_one(index):
            instance = cls()
            result, _logs, error = instance._generate_one(
                key, args['model_version'], args['thinking_level'], args['prompt'],
                args['aspect_ratio'], args['resolution'], args['seed'], args.get('image'))
            return result, error
    else:
        count = int(args.get('num_images', 1))
        model_id, cfg, quality, size = obj.prepare_model(
            args.get('model') or obj.DEFAULT_MODEL, args['quality'], args['size'])
        endpoint = obj.API_ENDPOINTS['image_edit' if args.get('input_image') is not None else 'text_to_image']
        headers = {'Authorization': f'Bearer {key}'}
        if endpoint == obj.API_ENDPOINTS['text_to_image']:
            headers['Content-Type'] = 'application/json'
        body, files = obj.build_request(endpoint, model_id, cfg, args['prompt'], size,
            quality, args['output_format'], args['output_compression'], args['background'],
            args.get('input_image'), args.get('mask_image'))
        def request_one(index):
            instance = cls()
            result, _usage, error, _elapsed = instance.send_single_request(
                endpoint, headers, body, files, instance.DEFAULT_TIMEOUT)
            return result, error
    if not 1 <= count <= 9:
        raise ValueError('Image count must be 1..9')
    return count, request_one, [key]


def save_images(tensor, file_prefix, index, output_dir):
    import numpy as np
    from PIL import Image
    directory = Path(output_dir) / 'api_immediate'
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    sequence = index + 1
    for frame in tensor:
        array = np.clip(frame.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        # Exclusive creation prevents overwrites, including short-ID collisions and
        # concurrent requests returning more than one image.
        while True:
            filename = f'{file_prefix}_{sequence}.png'
            sequence += 1
            try:
                stream = (directory / filename).open('xb')
                break
            except FileExistsError:
                continue
        with stream:
            Image.fromarray(array).save(stream, format='PNG')
        results.append({'filename': filename, 'subfolder': 'api_immediate', 'type': 'output'})
    return results
