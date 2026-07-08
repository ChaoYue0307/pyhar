"""Model backends. Real providers lazy-import their SDKs on construction, so
importing this module never requires anthropic/openai to be installed.

    from pyhar.models import AnthropicModel, OpenAIModel, OllamaModel, EchoModel

Bring your own model by implementing the tiny ``pyhar.Model`` protocol.
"""
from .anthropic import AnthropicModel
from .echo import EchoModel
from .ollama import OllamaModel
from .openai import OpenAICompatibleModel, OpenAIModel

__all__ = [
    "EchoModel",
    "AnthropicModel",
    "OpenAIModel",
    "OpenAICompatibleModel",
    "OllamaModel",
]
