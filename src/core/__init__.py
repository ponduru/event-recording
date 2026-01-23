"""Core module initialization."""

from src.core.domain import (
    Domain,
    DomainConfig,
    DomainRegistry,
    EventType,
    register_domain,
)

__all__ = [
    "Domain",
    "DomainConfig",
    "DomainRegistry",
    "EventType",
    "register_domain",
]
