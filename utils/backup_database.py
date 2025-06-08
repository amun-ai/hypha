import asyncio
import argparse
from collections import defaultdict, deque
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from typing import List, Optional
from sqlalchemy import Column, JSON, UniqueConstraint, select
from sqlmodel import SQLModel, Field
from datetime import datetime


# --- Model ---
class ArtifactModel(SQLModel, table=True):
    __tablename__ = "artifacts"
    id: str = Field(primary_key=True)
    type: Optional[str] = Field(default=None)
    workspace: str = Field(index=True)
    parent_id: Optional[str] = Field(default=None)
    alias: Optional[str] = Field(default=None)
    manifest: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    staging: Optional[list] = Field(default=None, sa_column=Column(JSON))
    download_count: float = Field(default=0.0)
    view_count: float = Field(default=0.0)
    file_count: int = Field(default=0)
    created_at: int = Field()
    created_by: Optional[str] = Field(default=None)
    last_modified: int = Field()
    versions: Optional[list] = Field(default=None, sa_column=Column(JSON))
    config: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    secrets: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    __table_args__ = (
        UniqueConstraint("workspace", "alias", name="_workspace_alias_uc"),
    )


def setup_engines(pg_uri: str, sqlite_prefix: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sqlite_path = f"./{sqlite_prefix}-{timestamp}.db"

    print(f"📦 Backup SQLite DB will be stored at: {sqlite_path}")

    # Globalize engine/session variables
    global pg_engine, pg_session_maker
    global sqlite_engine, sqlite_session_maker

    pg_engine = create_async_engine(pg_uri, echo=False)
    pg_session_maker = async_sessionmaker(
        pg_engine, class_=AsyncSession, expire_on_commit=False
    )

    sqlite_engine = create_async_engine(
        f"sqlite+aiosqlite:///{sqlite_path}", echo=False
    )
    sqlite_session_maker = async_sessionmaker(
        sqlite_engine, class_=AsyncSession, expire_on_commit=False
    )


def build_dependency_graph(artifacts):
    graph = defaultdict(list)
    indegree = defaultdict(int)
    id_map = {a.id: a for a in artifacts}

    for a in artifacts:
        if a.parent_id and a.parent_id in id_map:
            graph[a.parent_id].append(a.id)
            indegree[a.id] += 1
        else:
            indegree[a.id] = indegree.get(a.id, 0)

    return graph, indegree, id_map


def topological_sort_artifacts(artifacts):
    graph, indegree, id_map = build_dependency_graph(artifacts)
    queue = deque([aid for aid, deg in indegree.items() if deg == 0])
    ordered = []

    while queue:
        aid = queue.popleft()
        ordered.append(id_map[aid])
        for neighbor in graph[aid]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return ordered


def clean_artifact(a: ArtifactModel) -> ArtifactModel:
    return ArtifactModel(
        id=a.id,
        type=a.type,
        workspace=a.workspace,
        parent_id=a.parent_id,
        alias=a.alias,
        manifest=a.manifest,
        staging=a.staging,
        download_count=a.download_count,
        view_count=a.view_count,
        file_count=a.file_count,
        created_at=a.created_at,
        created_by=a.created_by,
        last_modified=a.last_modified,
        versions=a.versions,
        config=a.config,
        secrets=a.secrets,
    )


async def init_sqlite_db():
    async with sqlite_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def backup_artifacts_to_sqlite():
    async with pg_session_maker() as pg_session:
        result = await pg_session.execute(select(ArtifactModel))
        all_artifacts = result.scalars().all()

    ordered = topological_sort_artifacts(all_artifacts)
    print(f"Backing up {len(ordered)} artifacts to SQLite...")

    async with sqlite_session_maker() as sqlite_session:
        async with sqlite_session.begin():
            for a in ordered:
                try:
                    clean = clean_artifact(a)
                    sqlite_session.add(clean)
                except Exception as e:
                    print(f"⚠️ Failed to backup {a.id} ({a.alias}): {e}")
        await sqlite_session.commit()

    print("✅ Backup to SQLite completed.")


async def verify_migration_counts():
    async with pg_session_maker() as pg_session:
        pg_result = await pg_session.execute(select(ArtifactModel))
        pg_count = len(pg_result.scalars().all())

    async with sqlite_session_maker() as sqlite_session:
        sqlite_result = await sqlite_session.execute(select(ArtifactModel))
        sqlite_count = len(sqlite_result.scalars().all())

    print("📊 Migration Verification Summary")
    print(f" - PostgreSQL artifact count: {pg_count}")
    print(f" - SQLite artifact count:     {sqlite_count}")

    if pg_count == sqlite_count:
        print("✅ Verification passed: Record counts match.")
    else:
        print("❌ Verification failed: Mismatched record counts.")


async def main(pg_uri: str, sqlite_prefix: str):
    setup_engines(pg_uri, sqlite_prefix)
    await init_sqlite_db()
    await backup_artifacts_to_sqlite()
    await verify_migration_counts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backup PostgreSQL artifacts table to SQLite."
    )
    parser.add_argument(
        "--database-uri",
        required=True,
        help="PostgreSQL database URI (e.g., postgresql+asyncpg://user:pass@host/db)",
    )
    parser.add_argument(
        "--database-name",
        default="hypha-app-backup",
        help="Prefix for the SQLite backup file (default: hypha-app-backup)",
    )

    args = parser.parse_args()
    asyncio.run(main(args.database_uri, args.database_name))
