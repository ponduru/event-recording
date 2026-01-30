"""Database configuration."""

import os
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "prismata"
    username: str = "prismata"
    password: str = ""
    ssl_mode: str = "prefer"

    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800  # 30 minutes

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        return cls(
            host=os.getenv("PRISMATA_DB_HOST", "localhost"),
            port=int(os.getenv("PRISMATA_DB_PORT", "5432")),
            database=os.getenv("PRISMATA_DB_NAME", "prismata"),
            username=os.getenv("PRISMATA_DB_USER", "prismata"),
            password=os.getenv("PRISMATA_DB_PASSWORD", ""),
            ssl_mode=os.getenv("PRISMATA_DB_SSL_MODE", "prefer"),
            pool_size=int(os.getenv("PRISMATA_DB_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("PRISMATA_DB_MAX_OVERFLOW", "10")),
            pool_timeout=int(os.getenv("PRISMATA_DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("PRISMATA_DB_POOL_RECYCLE", "1800")),
        )

    def get_url(self, driver: str = "postgresql+psycopg2") -> str:
        """Get SQLAlchemy database URL."""
        return (
            f"{driver}://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


def get_database_url(config: DatabaseConfig | None = None) -> str:
    """Get database URL from config or environment.

    Priority:
    1. Provided config
    2. DATABASE_URL environment variable
    3. Individual environment variables
    4. Local SQLite fallback for development
    """
    # Check for explicit DATABASE_URL first
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url

    # Check if PostgreSQL config is available
    if config is None:
        config = DatabaseConfig.from_env()

    if config.password:
        return config.get_url()

    # Fallback to SQLite for local development
    sqlite_path = os.getenv("PRISMATA_SQLITE_PATH", ".cache/prismata/prismata.db")
    return f"sqlite:///{sqlite_path}"
