import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'api_immediate/__init__.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeRegistrationTests(unittest.TestCase):
    def test_two_bundles_share_manager_and_register_routes_once(self):
        instance = types.SimpleNamespace(routes=web.RouteTableDef())
        server = types.ModuleType('server')
        server.PromptServer = types.SimpleNamespace(instance=instance)
        nodes = types.ModuleType('nodes')
        nodes.NODE_CLASS_MAPPINGS = {}
        paths = types.ModuleType('folder_paths')
        paths.get_output_directory = lambda: 'unused-in-test'
        before = set(sys.modules)
        try:
            with patch.dict(sys.modules, {'server': server, 'nodes': nodes, 'folder_paths': paths}):
                first = load('bundle_a').install()
                second = load('bundle_b').install()
                self.assertIs(first, second)
                self.assertIs(first.manager, second.manager)
                routes = [(route.method, route.path) for route in instance.routes]
                self.assertEqual(len(routes), 3)
                self.assertEqual(len(set(routes)), 3)
        finally:
            for name in set(sys.modules) - before:
                if name.startswith('_comfy_api_immediate_runtime_v1_'):
                    del sys.modules[name]


if __name__ == '__main__':
    unittest.main()
