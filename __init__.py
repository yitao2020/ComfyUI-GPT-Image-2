from .gpt_image2_node import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    WEB_DIRECTORY,
)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

# Bundled independent API tasks; install() is idempotent across both plugins.
from .api_immediate import install as _install_immediate
_install_immediate()
