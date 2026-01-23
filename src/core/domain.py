"""Prismata: Core domain abstraction for multi-domain event intelligence."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type

import torch.nn as nn
from torch.utils.data import Dataset


class EventType(Enum):
    """Types of events that can be detected across domains."""

    # Cricket events
    CRICKET_DELIVERY = "cricket_delivery"

    # Soccer events
    SOCCER_GOAL = "soccer_goal"
    SOCCER_PENALTY = "soccer_penalty"
    SOCCER_CORNER_KICK = "soccer_corner_kick"
    SOCCER_FREE_KICK = "soccer_free_kick"

    # Warehouse events
    WAREHOUSE_PACKAGE_PICKUP = "warehouse_package_pickup"
    WAREHOUSE_FORKLIFT_MOVEMENT = "warehouse_forklift_movement"
    WAREHOUSE_SAFETY_VIOLATION = "warehouse_safety_violation"

    # Factory events
    FACTORY_ASSEMBLY_COMPLETE = "factory_assembly_complete"
    FACTORY_QUALITY_ISSUE = "factory_quality_issue"
    FACTORY_MACHINE_IDLE = "factory_machine_idle"


@dataclass
class DomainConfig:
    """Base configuration for a domain."""

    name: str
    event_types: List[EventType]
    description: str
    default_model_params: Dict[str, Any]


class Domain(ABC):
    """Abstract base class for event detection domains.

    Each domain (e.g., cricket, soccer, warehouse) should implement this interface
    to provide domain-specific models, datasets, and configurations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Domain name (e.g., 'cricket', 'soccer', 'warehouse')."""
        pass

    @property
    @abstractmethod
    def event_types(self) -> List[EventType]:
        """List of event types this domain can detect."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the domain."""
        pass

    @abstractmethod
    def create_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create a model for this domain.

        Args:
            config: Model configuration parameters.

        Returns:
            PyTorch model instance.
        """
        pass

    @abstractmethod
    def create_dataset(
        self,
        labels_dir: str,
        videos_dir: str,
        **kwargs,
    ) -> Dataset:
        """Create a training dataset for this domain.

        Args:
            labels_dir: Directory containing label files.
            videos_dir: Directory containing video files.
            **kwargs: Additional dataset parameters.

        Returns:
            PyTorch Dataset instance.
        """
        pass

    @abstractmethod
    def create_inference_dataset(
        self,
        video_path: str,
        **kwargs,
    ) -> Dataset:
        """Create an inference dataset for this domain.

        Args:
            video_path: Path to video file.
            **kwargs: Additional dataset parameters.

        Returns:
            PyTorch Dataset instance for inference.
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for this domain.

        Returns:
            Dictionary of default parameters.
        """
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate domain-specific configuration.

        Args:
            config: Configuration to validate.

        Returns:
            True if valid, raises ValueError otherwise.
        """
        # Default implementation - domains can override
        return True


class DomainRegistry:
    """Registry for managing available domains."""

    _domains: Dict[str, Type[Domain]] = {}

    @classmethod
    def register(cls, domain_class: Type[Domain]) -> None:
        """Register a domain.

        Args:
            domain_class: Domain class to register.
        """
        # Instantiate to get the name
        instance = domain_class()
        cls._domains[instance.name] = domain_class

    @classmethod
    def get(cls, name: str) -> Domain:
        """Get a domain by name.

        Args:
            name: Domain name.

        Returns:
            Domain instance.

        Raises:
            ValueError: If domain not found.
        """
        if name not in cls._domains:
            available = ", ".join(cls._domains.keys())
            raise ValueError(
                f"Domain '{name}' not found. Available domains: {available}"
            )
        return cls._domains[name]()

    @classmethod
    def list_domains(cls) -> List[str]:
        """List all registered domain names.

        Returns:
            List of domain names.
        """
        return list(cls._domains.keys())

    @classmethod
    def get_all(cls) -> Dict[str, Domain]:
        """Get all registered domains.

        Returns:
            Dictionary mapping domain names to instances.
        """
        return {name: domain_cls() for name, domain_cls in cls._domains.items()}


def register_domain(domain_class: Type[Domain]) -> Type[Domain]:
    """Decorator to register a domain.

    Usage:
        @register_domain
        class CricketDomain(Domain):
            ...
    """
    DomainRegistry.register(domain_class)
    return domain_class
