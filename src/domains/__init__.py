"""Domains module initialization.

This module auto-registers all available domains when imported.
"""

# Import all domains to trigger registration
from src.domains.cricket import CricketDomain  # noqa: F401
from src.domains.soccer import SoccerDomain  # noqa: F401
from src.domains.warehouse import WarehouseDomain  # noqa: F401

__all__ = ["CricketDomain", "SoccerDomain", "WarehouseDomain"]
