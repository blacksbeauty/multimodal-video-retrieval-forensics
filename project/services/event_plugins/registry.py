from __future__ import annotations

from typing import Dict, Type

from services.event_plugins.base import EventPluginBase


PLUGINS: Dict[str, Type[EventPluginBase]] = {}


def register(plugin_name: str, plugin_cls: Type[EventPluginBase]) -> None:
    """Register an event plugin class by name."""
    PLUGINS[plugin_name] = plugin_cls


def get(plugin_name: str) -> Type[EventPluginBase] | None:
    """Resolve a registered event plugin by name."""
    return PLUGINS.get(plugin_name)


def list_plugins() -> list[str]:
    """List registered plugin names."""
    return sorted(PLUGINS.keys())
