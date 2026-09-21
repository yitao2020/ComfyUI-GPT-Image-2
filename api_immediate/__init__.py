"""Bundled runtime shared by both image plugins; routes register once per server."""
import importlib.util
from pathlib import Path
import sys

RUNTIME_ABI = 1


def install():
    from server import PromptServer
    server = PromptServer.instance
    attribute = "_api_immediate_runtime_v1"
    existing = getattr(server, attribute, None)
    if existing is not None:
        return existing
    name = f"_comfy_api_immediate_runtime_v1_{id(server)}"
    directory = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        name, directory / "runtime.py", submodule_search_locations=[str(directory)])
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[name] = runtime
    try:
        spec.loader.exec_module(runtime)
    except Exception:
        sys.modules.pop(name, None)
        raise
    setattr(server, attribute, runtime)
    return runtime
