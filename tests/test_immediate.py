import asyncio
import base64
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import requests

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load('immediate_test_core', ROOT / 'api_immediate/core.py')
backend = load('immediate_test_backend', ROOT / 'api_immediate/backend.py')


def payload(prompt='first'):
    return {'target': '1', 'prompt': {'1': {'class_type': 'NanoBananaPro',
                                           'inputs': {'prompt': prompt}}}}


def wait_for(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for expected state')


class SchedulingTests(unittest.TestCase):
    def test_new_submission_finishes_while_first_is_blocked_and_snapshot_isolated(self):
        started, release = threading.Event(), threading.Event()
        observed = []
        def prepare(data):
            prompt = data['prompt']['1']['inputs']['prompt']
            def call(index):
                observed.append(prompt)
                if prompt == 'first':
                    started.set()
                    release.wait(4)
                return prompt, None
            return 1, call, []
        manager = core.JobManager(prepare, lambda image, job, index: [image])
        original = payload()
        first = manager.submit('owner', original)
        self.assertTrue(started.wait(2))
        original['prompt']['1']['inputs']['prompt'] = 'mutated'
        second = manager.submit('owner', payload('second'))
        try:
            wait_for(lambda: manager.jobs[second]['status'] == 'completed')
            self.assertEqual(manager.jobs[first]['status'], 'running')
            self.assertEqual(observed, ['first', 'second'])
            self.assertEqual(manager.list('other'), [])
            self.assertNotIn('owner', manager.list('owner')[0])
        finally:
            release.set()
        wait_for(lambda: manager.jobs[first]['status'] == 'completed')

    def test_all_images_start_together_and_cancel_discards_late_results(self):
        barrier = threading.Barrier(4)
        release = threading.Event()
        saved = []
        def call(index):
            barrier.wait(3)
            release.wait(3)
            return index, None
        manager = core.JobManager(lambda _: (3, call, []), lambda *args: saved.append(args))
        job = manager.submit('owner', payload())
        try:
            barrier.wait(3)
            self.assertFalse(manager.cancel('other', job))
            self.assertTrue(manager.cancel('owner', job))
        finally:
            release.set()
        time.sleep(.1)
        self.assertEqual(manager.jobs[job]['status'], 'cancelled')
        self.assertEqual(saved, [])

    def test_partial_failures_redact_secrets(self):
        def call(index):
            return ('image', None) if index == 0 else (None, 'Rejected secret-token')
        manager = core.JobManager(lambda _: (2, call, ['secret-token']), lambda *args: ['saved'])
        job = manager.submit('owner', payload())
        wait_for(lambda: manager.jobs[job]['status'] == 'partial')
        self.assertNotIn('secret-token', str(manager.list('owner')))

    def test_quota_error_explained_and_node_identity_preserved(self):
        manager = core.JobManager(lambda _: (1, lambda i: (None, '403 insufficient_user_quota'), []), lambda *args: [])
        data = payload()
        data['client_ref'] = 'node-stable-id'
        job = manager.submit('owner', data)
        wait_for(lambda: manager.jobs[job]['status'] == 'failed')
        state = manager.list('owner')[0]
        self.assertEqual(state['client_ref'], 'node-stable-id')
        self.assertIn('额度不足', state['items'][0]['error'])


class RealAdapterTests(unittest.TestCase):
    def setUp(self):
        blocker = patch.object(requests.sessions.Session, "request", side_effect=AssertionError("Live HTTP forbidden in tests"))
        blocker.start()
        self.addCleanup(blocker.stop)

    @classmethod
    def setUpClass(cls):
        from PIL import Image
        image = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(image, format='PNG')
        cls.b64 = base64.b64encode(image.getvalue()).decode()
        cls.nano = cls.gpt = None
        cls.registry = {}
        if (ROOT / 'nanobanana_pro_node.py').exists():
            cls.nano = load('test_actual_nano', ROOT / 'nanobanana_pro_node.py')
            cls.registry.update(cls.nano.NODE_CLASS_MAPPINGS)
        if (ROOT / 'gpt_image2_node.py').exists():
            package = types.ModuleType('test_actual_gpt')
            package.__path__ = [str(ROOT)]
            sys.modules[package.__name__] = package
            fake_utils = types.ModuleType('comfy.utils')
            fake_utils.ProgressBar = object
            fake_server = types.ModuleType('server')
            with patch.dict(sys.modules, {'comfy.utils': fake_utils, 'server': fake_server}):
                cls.gpt = load('test_actual_gpt.gpt_image2_node', ROOT / 'gpt_image2_node.py')
            cls.registry.update(cls.gpt.NODE_CLASS_MAPPINGS)

    def make_payload(self, kind):
        values = {}
        schema = self.registry[kind].INPUT_TYPES()
        for name, spec in schema['required'].items():
            opts = spec[1] if len(spec) > 1 else {}
            values[name] = opts.get('default', spec[0][0] if isinstance(spec[0], list) else '')
        values['api_key'] = 'test-key-never-sent-to-network'
        values['prompt'] = 'Test image'
        return {'target': '1', 'prompt': {'1': {'class_type': kind, 'inputs': values}}}

    def test_actual_providers_mock_http_save_distinct_images_and_report_http_errors(self):
        from unittest.mock import Mock
        for kind in self.registry:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as output:
                data = self.make_payload(kind)
                data['prompt']['1']['inputs']['image_count' if kind == 'NanoBananaPro' else 'num_images'] = 2 if kind == 'NanoBananaPro' else '2'
                backend.validate(data, self.registry)
                count, call, secrets = backend.prepare(data, self.registry)
                response = Mock(status_code=200)
                response.json.return_value = ({'candidates': [{'content': {'parts': [
                    {'inlineData': {'mimeType': 'image/png', 'data': self.b64}}]}}]}
                    if kind == 'NanoBananaPro' else {'data': [{'b64_json': self.b64}]})
                with patch.object((self.nano if kind == 'NanoBananaPro' else self.gpt).requests, 'post', return_value=response) as post:
                    tensor, error = call(0)
                    self.assertFalse(error)
                    self.assertEqual(tuple(tensor.shape), (1, 8, 8, 3))
                    self.assertEqual(post.call_count, 1)
                    self.assertEqual(count, 2)
                    first = backend.save_images(tensor, 'a' * 32, 0, output)
                    second = backend.save_images(tensor, 'b' * 32, 0, output)
                    self.assertNotEqual(first, second)
                    self.assertEqual(len(list(Path(output).rglob('*.png'))), 2)
                    response.status_code = 429
                    response.text = 'rate limited'
                    response.json.return_value = {'error': 'rate limited'}
                    tensor, error = call(1)
                    self.assertIsNone(tensor)
                    self.assertIn('429', error)

    @unittest.skipUnless((ROOT / 'nanobanana_pro_node.py').exists(), 'Nano adapter tests belong to Nano repository')
    def test_upstream_image_snapshot_and_unsupported_graph_rejected(self):
        import torch
        class Source:
            RETURN_TYPES = ('IMAGE',)
            FUNCTION = 'load'
            @classmethod
            def INPUT_TYPES(cls): return {'required': {'image': ('STRING',)}}
            def load(self, image): return (torch.zeros(1, 8, 8, 3),)
        data = self.make_payload('NanoBananaPro')
        data['prompt']['1']['inputs']['image'] = ['2', 0]
        data['prompt']['2'] = {'class_type': 'LoadImage', 'inputs': {'image': 'ref.png'}}
        registry = {**self.registry, 'LoadImage': Source}
        clean = backend.validate(data, registry)
        with patch.object(self.nano.NanoBananaProImageGenerator, '_generate_one', return_value=('result', [], None)) as send:
            count, call, _ = backend.prepare(clean, registry)
            call(0)
            self.assertEqual(tuple(send.call_args.args[-1].shape), (1, 8, 8, 3))
        data['prompt']['2']['class_type'] = 'KSampler'
        with self.assertRaisesRegex(ValueError, 'KSampler'):
            backend.validate(data, registry)

    @unittest.skipUnless((ROOT / 'nanobanana_pro_node.py').exists(), 'Nano adapter tests belong to Nano repository')
    def test_stale_workflow_schema_rejected_before_api_call(self):
        data = self.make_payload('NanoBananaPro')
        data['prompt']['1']['inputs']['model'] = 'old-model-widget'
        with self.assertRaisesRegex(ValueError, '参数与安装版本不匹配'):
            backend.validate(data, self.registry)
        del data['prompt']['1']['inputs']['model']
        data['prompt']['1']['inputs']['model_version'] = 'prompt in wrong widget position'
        with self.assertRaisesRegex(ValueError, 'model_version 参数无效'):
            backend.validate(data, self.registry)

    @unittest.skipUnless((ROOT / 'gpt_image2_node.py').exists(), 'GPT adapter test belongs to GPT repository')
    def test_original_gpt_node_quota_hint_is_not_content_moderation(self):
        from unittest.mock import Mock
        instance = self.gpt.GPTImage2Node()
        args = self.make_payload('GPTImage2')['prompt']['1']['inputs']
        with patch.object(self.gpt, 'ProgressBar', return_value=Mock()), \
             patch.object(instance, 'get_api_key', return_value='mock-key'), \
             patch.object(instance, 'send_single_request', return_value=(None, {}, '状态码 403 insufficient_user_quota', 0)):
            _, info = instance.generate_image(**args)
        self.assertIn('API 额度不足', info)
        self.assertNotIn('内容审核拦截', info)

    @unittest.skipUnless((ROOT / 'nanobanana_pro_node.py').exists(), 'Nano adapter tests belong to Nano repository')
    def test_dynamic_batch_does_not_mix_source_input_names_into_parent(self):
        import torch
        class Source:
            RETURN_TYPES = ('IMAGE',)
            FUNCTION = 'load'
            @classmethod
            def INPUT_TYPES(cls): return {'required': {'image': ('STRING',)}}
            def load(self, image): return (torch.zeros(1, 8, 8, 3),)
        class Batch:
            RETURN_TYPES = ('IMAGE',)
            @classmethod
            def INPUT_TYPES(cls): return {'required': {'images': ('COMFY_AUTOGROW_V3',)}}
            @classmethod
            def execute(cls, images):
                assert list(images) == ['image0', 'image1']
                return types.SimpleNamespace(result=(torch.cat(list(images.values())),))
        data = self.make_payload('NanoBananaPro')
        data['prompt']['1']['inputs']['image'] = ['3', 0]
        data['prompt']['2'] = {'class_type': 'LoadImage', 'inputs': {'image': 'ref.png'}}
        data['prompt']['3'] = {'class_type': 'BatchImagesNode', 'inputs': {
            'images.image1': ['2', 0], 'images.image0': ['2', 0]}}
        registry = {**self.registry, 'LoadImage': Source, 'BatchImagesNode': Batch}
        clean = backend.validate(data, registry)
        with patch.object(self.nano.NanoBananaProImageGenerator, '_generate_one', return_value=('result', [], None)) as send:
            _, call, _ = backend.prepare(clean, registry)
            call(0)
            self.assertEqual(tuple(send.call_args.args[-1].shape), (2, 8, 8, 3))


if __name__ == '__main__':
    unittest.main()
