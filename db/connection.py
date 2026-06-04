"""
Database connection pool — shared across the FastAPI app.

Uses SQLAlchemy async engine backed by asyncpg.
Call init_db() once at startup to create tables and enable TimescaleDB extensions.
"""
from __future__ import annotations
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import settings

# Convert postgresql:// → postgresql+asyncpg://
_async_url = settings.database_url.replace(
    "postgresql://", "postgresql+asyncpg://"
)

engine = create_async_engine(
    _async_url,
    pool_size=5,
    max_overflow=10,
    echo=False,          # set True to log all SQL
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
