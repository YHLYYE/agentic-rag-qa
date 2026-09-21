"""Patch ragas 0.4.3's broken `langchain_community.*.vertexai` imports.

ragas 0.4.3 unconditionally imports vertexai chat/llm/embedding classes from
`langchain_community.*.vertexai`, but those modules were moved to
`langchain_google_vertexai` in newer langchain-community. We never use
VertexAI (we use DeepSeek), so intercept ALL such imports with dummy classes.
"""
import importlib.abc
import importlib.util
import sys
import types

_DUMMY_NAMES = (
    "ChatVertexAI", "VertexAI", "VertexAIChat", "VertexAIEmbeddings",
    "VertexAIModelGarden",
)


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        for name in _DUMMY_NAMES:
            setattr(mod, name, type(name, (), {}))
        return mod

    def exec_module(self, module):
        pass


class _VertexAIFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("langchain_community") and ".vertexai" in fullname:
            return importlib.util.spec_from_loader(fullname, _StubLoader())
        return None


def patch() -> None:
    if not any(isinstance(f, _VertexAIFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _VertexAIFinder())
