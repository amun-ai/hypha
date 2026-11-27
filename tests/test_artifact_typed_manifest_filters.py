"""`mf-0018` — a BOOLEAN (or numeric) manifest filter must be queryable, not a server traceback.

Reported 2026-11-26 by the agent teams running on Mycolab, against hypha.aicell.io:

    filters={'manifest': {'needs_maintainer': True}}    -> FAILED
    filters={'manifest': {'needs_maintainer': 'yes'}}   -> OK, 1 hit

    asyncpg.exceptions.UndefinedFunctionError:
      operator does not exist: text = boolean
    HINT: No operator matches the given name and argument types.
          You might need to add explicit type casts.

    ... AND manifest ->> $3::VARCHAR = $4::BOOLEAN ...
    parameters: (..., 'needs_maintainer', True, 200, 0)

THE CAUSE, which the reporter diagnosed correctly: `manifest ->> :k` always yields **text**,
while the comparand was bound with its *Python* type. Postgres has no `text = boolean`, so the
whole query is refused.

WHAT IT COST, which is the part that makes this a bug and not a papercut: *"Any boolean field in a
manifest is unfilterable. We hit it designing this very collection: the intended convention used
two boolean flags so each side could find the rows it owed an answer on, and it had to be
redesigned as a single string field (`awaiting`) because the boolean form cannot be queried."*
A data model was changed to route around a type coercion.

⚠ AND NUMBERS WERE BROKEN BY THE SAME OPERATOR, unreported because nobody had tried: there is no
`text = integer` either. Both are covered below, because both are one defect.

Two levels, deliberately:

  * the MAPPING is a unit test — no server, no database. The defect is a type coercion, and a
    coercion is exactly the thing a unit test can pin exhaustively (including the `bool`-is-an-`int`
    trap, which is the one way a careless fix gets this wrong).
  * the BEHAVIOUR is an end-to-end test through the real `artifact-manager` service, which is what
    proves the emitted SQL is accepted and selects the right rows. It runs against SQLite here
    (fast, and it is the path that must NOT change); `test_artifact.py` exercises Postgres in CI,
    which is the dialect that raised.
"""

import pytest
from hypha_rpc import connect_to_server

from hypha.artifact import ArtifactController

from . import SERVER_URL, SERVER_URL_SQLITE

pytestmark = pytest.mark.asyncio


def test_pg_text_renders_json_text_for_every_scalar_type():
    """The mapping Postgres' `->>` will be compared against.

    ⚠ `True` must become `'true'` and NOT `'1'`. In Python `isinstance(True, int)` is True, so a
    fix that checks `int` before `bool` renders every boolean as a number — which raises nothing,
    matches nothing, and is therefore strictly worse than the traceback it replaced.
    """
    render = ArtifactController._pg_text

    assert render("k", True) == "true"
    assert render("k", False) == "false"
    # The precise shape of the trap above, asserted rather than implied.
    assert render("k", True) != "1"
    assert render("k", False) not in ("0", "")

    assert render("k", 5) == "5"
    assert render("k", -3) == "-3"
    assert render("k", 1.5) == "1.5"

    # Strings pass through untouched — the case that always worked must keep working byte for byte.
    assert render("k", "yes") == "yes"
    assert render("k", "true") == "true"
    assert render("k", "") == ""


def test_pg_text_refuses_none_instead_of_matching_nothing():
    """`None` is refused, by name.

    `manifest ->> 'k'` is SQL NULL both when the key is absent and when its value is JSON null, so
    `= NULL` is never true. Binding it would answer "no rows" for every input, with exit 0 and no
    error — a plausible wrong answer, which is the failure mode this whole report is about. The
    reporter's own fallback ask was *"at minimum, reject a boolean filter with a message naming the
    field, so it is a rejected argument rather than a server traceback"*; this is that, for the one
    value we genuinely cannot express.
    """
    with pytest.raises(ValueError) as e:
        ArtifactController._pg_text("needs_maintainer", None)
    # The field name is the whole point of raising rather than returning — a refusal that does not
    # say WHICH filter was at fault is the same complaint one level down.
    assert "needs_maintainer" in str(e.value)


async def _drive_typed_filters(server_url, token, alias):
    """The reporter's reproduction, driven against whichever backend `server_url` names.

    ⚠ ONE BODY, TWO BACKENDS, and the split below is the whole point. See the two callers.
    """
    api = await connect_to_server(
        {"name": "typed-filter-client", "server_url": server_url, "token": token}
    )
    artifact_manager = await api.get_service("public/artifact-manager")

    collection = await artifact_manager.create(
        type="collection",
        alias=alias,
        manifest={"name": "Typed manifest filters"},
        config={"permissions": {"*": "r", "@": "rw+"}},
    )

    rows = {
        "bool-true": {"needs_maintainer": True, "count": 5},
        "bool-false": {"needs_maintainer": False, "count": 7},
        "string-yes": {"needs_maintainer": "yes", "count": 9},
    }
    for row_alias, manifest in rows.items():
        art = await artifact_manager.create(
            type="dataset",
            alias=f"{alias}-{row_alias}",
            parent_id=collection.id,
            manifest={"name": row_alias, **manifest},
            version="stage",
        )
        await artifact_manager.commit(artifact_id=art.id)

    async def names(manifest_filter):
        found = await artifact_manager.list(
            parent_id=collection.id, filters={"manifest": manifest_filter}, mode="AND"
        )
        return sorted(a["manifest"]["name"] for a in found)

    # THE REPORT: a boolean filter must select its row, and nothing else.
    assert await names({"needs_maintainer": True}) == ["bool-true"]
    # `False` is a value and not an absence — a fix that renders it falsy would select nothing here,
    # or (worse) everything.
    assert await names({"needs_maintainer": False}) == ["bool-false"]
    # THE CONTROL: the string path the reporter had to fall back to still works, unchanged.
    assert await names({"needs_maintainer": "yes"}) == ["string-yes"]
    # The same operator with no `text = integer` either.
    assert await names({"count": 7}) == ["bool-false"]

    # A boolean filter must not be satisfied by the STRING 'true' or vice versa — if it were, the
    # three assertions above could pass on a coercion that collapses the two, and the manifest
    # convention the reporter wanted would still be unusable.
    assert await names({"needs_maintainer": "True"}) == []


async def test_typed_manifest_filters_on_postgres_the_dialect_that_RAISED(
    minio_server, fastapi_server, test_user_token
):
    """THE CASE. Postgres is the dialect that refused `text = boolean`, so it is the only backend
    on which this test can fail for the reported reason.

    ⚠ Needs the `postgres_server` fixture, which is a docker container locally and a service
    container in GitHub Actions. On a box without docker it ERRORS rather than passing, which is
    the honest outcome: a case that could not run has not run. Do not convert it to a skip — a
    skipped case and a passing one are reported with the same exit code, and this is the only arm
    that measures the fix.
    """
    await _drive_typed_filters(SERVER_URL, test_user_token, "typed-filters-pg")


async def test_typed_manifest_filters_on_sqlite_is_the_CONTROL_not_the_case(
    minio_server, fastapi_server_sqlite, test_user_token
):
    """THE CONTROL, and it is labelled as one because it **passes with the fix removed**.

    Measured 2026-11-27: deleting the `_pg_text` call from `_build_manifest_condition` — the
    reported defect, restored — leaves this case GREEN. That is not a weak test, it is a test of a
    different thing: `json_extract` returns a NATIVE SQL value (1/0 for a JSON boolean, a number
    for a number), so SQLite never takes the `is_pg` branch the coercion lives in and cannot
    observe it.

    What it DOES prove is the half that a careless fix breaks: coercing on both branches would turn
    these working filters into ones that silently match nothing. Keeping it, and naming it a
    control, is what stops the next reader taking a green SQLite run as evidence about Postgres.
    """
    await _drive_typed_filters(SERVER_URL_SQLITE, test_user_token, "typed-filters-sqlite")
