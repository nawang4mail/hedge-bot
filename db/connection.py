"""
Database connection pool — shared across the FastAPI app.

Uses SQLAlchemy async engine backed by asyncpg.
Call init_db() once at startup to create tables and enable TimescaleDB extensions.
"""
from __future__ import annotations
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool
from config import settings

# Convert postgresql:// → postgresql+asyncpg://
_async_url = settings.database_url.replace(
    "postgresql://", "postgresql+asyncpg://"
)

# NullPool: no connection pooling. Each AsyncSessionLocal() opens a fresh
# asyncpg connection in whatever event loop is current. Required because the
# pipeline runs via run_in_executor (thread-local event loop) while FastAPI
# routes run in the main loop — pooled connections bind to one loop and fail
# if reused from the other.
engine = create_async_engine(
    _async_url,
    poolclass=NullPool,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency — yields an async session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """
    Called once at app startup.
    Creates all tables and converts time-series tables to TimescaleDB hypertables.
    """
    # Import all models so SQLAlchemy registers them before create_all
    import db.models           # noqa: F401
    import db.training_models  # noqa: F401
    import db.insider_models   # noqa: F401

    from sqlalchemy import text as _text
    async with engine.begin() as conn:
        # Enable TimescaleDB extension
        await conn.execute(_text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

        # Convert time-series tables to hypertables (safe to call multiple times)
        for table, col in [("ohlcv", "ts"), ("ohlcv_hourly", "ts")]:
            await conn.execute(_text(f"""
                SELECT create_hypertable(
                    '{table}', '{col}',
                    if_not_exists => TRUE,
                    migrate_data  => TRUE
                );
            """))
