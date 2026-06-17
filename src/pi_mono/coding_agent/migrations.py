"""One-time migrations that run on startup (stub for Python port)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MigrationResult:
    migrated_auth_providers: list[str]
    deprecation_warnings: list[str]


def run_migrations(cwd: str) -> MigrationResult:
    """Run startup migrations. Python port returns empty diagnostics for now."""
    del cwd
    return MigrationResult(migrated_auth_providers=[], deprecation_warnings=[])
