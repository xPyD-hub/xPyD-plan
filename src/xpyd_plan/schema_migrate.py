"""Benchmark schema versioning and migration.

Detect benchmark JSON schema version and migrate between versions.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class SchemaVersion(BaseModel):
    """Represents a schema version with comparison support."""

    major: int = Field(..., ge=1, description="Major version number")
    minor: int = Field(0, ge=0, description="Minor version number")

    @classmethod
    def parse(cls, version_str: str) -> SchemaVersion:
        """Parse a version string like '1', '1.0', '2', '2.0'.

        Args:
            version_str: Version string to parse.

        Returns:
            Parsed SchemaVersion.

        Raises:
            ValueError: If the version string is malformed.
        """
        parts = str(version_str).strip().split(".")
        if len(parts) == 1:
            return cls(major=int(parts[0]))
        if len(parts) == 2:
            return cls(major=int(parts[0]), minor=int(parts[1]))
        raise ValueError(f"Invalid schema version: {version_str!r}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return self.major == other.major and self.minor == other.minor

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (self.major, self.minor) <= (other.major, other.minor)

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (self.major, self.minor) > (other.major, other.minor)

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, SchemaVersion):
            return NotImplemented
        return (self.major, self.minor) >= (other.major, other.minor)

    def __hash__(self) -> int:
        return hash((self.major, self.minor))


class MigrationResult(BaseModel):
    """Result of a schema migration operation."""

    source_version: str = Field(..., description="Original schema version")
    target_version: str = Field(..., description="Target schema version after migration")
    migrated: bool = Field(..., description="Whether migration was performed")
    changes: list[str] = Field(default_factory=list, description="List of changes applied")
    data: dict[str, Any] = Field(..., description="Migrated benchmark data")


# Current latest version
LATEST_VERSION = SchemaVersion(major=2, minor=0)

# All known versions
KNOWN_VERSIONS = {
    SchemaVersion(major=1, minor=0),
    SchemaVersion(major=2, minor=0),
}


def detect_version(data: dict[str, Any]) -> SchemaVersion:
    """Detect the schema version of benchmark data.

    Args:
        data: Parsed benchmark JSON data.

    Returns:
        Detected SchemaVersion.

    Raises:
        ValueError: If the data format is unrecognized.
    """
    if "schema_version" in data:
        return SchemaVersion.parse(str(data["schema_version"]))

    # Native format without explicit version → v1.0
    if "metadata" in data and "requests" in data:
        return SchemaVersion(major=1, minor=0)

    # xpyd-bench format without explicit version → v1.0
    if "bench_config" in data and "results" in data:
        return SchemaVersion(major=1, minor=0)

    raise ValueError(
        "Cannot detect schema version. Expected 'schema_version' field, "
        "or recognized format with 'metadata'+'requests' or 'bench_config'+'results'."
    )


def _migrate_v1_to_v2(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Migrate v1 data to v2 format.

    v2 adds:
    - schema_version field at top level
    - metadata.run_id (UUID if not present)
    - metadata.schema_version

    Args:
        data: v1 benchmark data.

    Returns:
        Tuple of (migrated data, list of change descriptions).
    """
    result = deepcopy(data)
    changes: list[str] = []

    # Add top-level schema_version
    result["schema_version"] = "2.0"
    changes.append("Added top-level 'schema_version': '2.0'")

    # Add run_id to metadata if present
    if "metadata" in result:
        if "run_id" not in result["metadata"]:
            result["metadata"]["run_id"] = str(uuid.uuid4())
            changes.append("Added 'metadata.run_id' (generated UUID)")
        if "schema_version" not in result["metadata"]:
            result["metadata"]["schema_version"] = "2.0"
            changes.append("Added 'metadata.schema_version': '2.0'")

    return result, changes


class SchemaMigrator:
    """Detect and migrate benchmark data between schema versions."""

    def __init__(self) -> None:
        self._migrations: dict[tuple[SchemaVersion, SchemaVersion], Any] = {
            (SchemaVersion(major=1, minor=0), SchemaVersion(major=2, minor=0)): _migrate_v1_to_v2,
        }

    def detect(self, data: dict[str, Any]) -> SchemaVersion:
        """Detect schema version of benchmark data.

        Args:
            data: Parsed benchmark JSON.

        Returns:
            Detected SchemaVersion.
        """
        return detect_version(data)

    def needs_migration(
        self,
        data: dict[str, Any],
        target_version: SchemaVersion | None = None,
    ) -> bool:
        """Check if data needs migration to reach target version.

        Args:
            data: Parsed benchmark JSON.
            target_version: Target version (default: LATEST_VERSION).

        Returns:
            True if migration is needed.
        """
        target = target_version or LATEST_VERSION
        current = self.detect(data)
        return current < target

    def migrate(
        self,
        data: dict[str, Any],
        target_version: SchemaVersion | None = None,
        dry_run: bool = False,
    ) -> MigrationResult:
        """Migrate benchmark data to target version.

        Args:
            data: Parsed benchmark JSON.
            target_version: Target version (default: LATEST_VERSION).
            dry_run: If True, compute changes but return original data.

        Returns:
            MigrationResult with migrated data and change log.

        Raises:
            ValueError: If migration path is not available.
        """
        target = target_version or LATEST_VERSION
        current = self.detect(data)

        if current == target:
            return MigrationResult(
                source_version=str(current),
                target_version=str(target),
                migrated=False,
                changes=[],
                data=deepcopy(data),
            )

        if current > target:
            raise ValueError(
                f"Cannot downgrade from {current} to {target}. "
                "Only forward migrations are supported."
            )

        # Find migration path
        migration_key = (current, target)
        if migration_key not in self._migrations:
            raise ValueError(
                f"No migration path from {current} to {target}. "
                f"Known migrations: {list(self._migrations.keys())}"
            )

        migrate_fn = self._migrations[migration_key]
        migrated_data, changes = migrate_fn(data)

        if dry_run:
            return MigrationResult(
                source_version=str(current),
                target_version=str(target),
                migrated=False,
                changes=changes,
                data=deepcopy(data),
            )

        return MigrationResult(
            source_version=str(current),
            target_version=str(target),
            migrated=True,
            changes=changes,
            data=migrated_data,
        )

    def migrate_file(
        self,
        path: str | Path,
        target_version: SchemaVersion | None = None,
        dry_run: bool = False,
        output_path: str | Path | None = None,
    ) -> MigrationResult:
        """Migrate a benchmark JSON file.

        Args:
            path: Path to benchmark JSON file.
            target_version: Target version (default: LATEST_VERSION).
            dry_run: If True, preview changes without writing.
            output_path: Where to write migrated file (default: overwrite in place).

        Returns:
            MigrationResult.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        result = self.migrate(data, target_version=target_version, dry_run=dry_run)

        if result.migrated and not dry_run:
            out = Path(output_path) if output_path else path
            with open(out, "w") as f:
                json.dump(result.data, f, indent=2)
                f.write("\n")

        return result


def migrate_schema(
    path: str | Path,
    target_version: str | None = None,
    dry_run: bool = False,
    output_path: str | Path | None = None,
) -> MigrationResult:
    """Programmatic API: migrate a benchmark file's schema.

    Args:
        path: Path to benchmark JSON file.
        target_version: Target version string (default: latest).
        dry_run: Preview changes without writing.
        output_path: Output file path (default: overwrite in place).

    Returns:
        MigrationResult.
    """
    migrator = SchemaMigrator()
    target = SchemaVersion.parse(target_version) if target_version else None
    return migrator.migrate_file(
        path, target_version=target, dry_run=dry_run, output_path=output_path
    )
