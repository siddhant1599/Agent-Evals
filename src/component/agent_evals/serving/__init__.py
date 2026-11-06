"""Serving: every backbone is reached through one client interface, selected by name in config."""

from .backbones import BackboneSpec, ModelRegistryError, load_model_registry, select_backbone
from .base import ChatClient, Completion, ServingError, endpoint_url, get_client
from .launch import VLLMServer, build_serve_command
from .training_model import load_training_model
from .vllm_client import VLLMChatClient

__all__ = [
    "BackboneSpec",
    "ChatClient",
    "Completion",
    "ModelRegistryError",
    "ServingError",
    "VLLMChatClient",
    "VLLMServer",
    "build_serve_command",
    "endpoint_url",
    "get_client",
    "load_model_registry",
    "load_training_model",
    "select_backbone",
]
