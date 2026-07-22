"""#0006 (item 1): --from-env must NOT silently clobber explicit CLI args.

`create_application`'s `--from-env` path previously did
`setattr(args, key, value)` for EVERY environment-derived arg, overwriting
explicitly-provided command-line flags (e.g. `--startup-functions=...`) with the
environment default — so a custom auth provider passed on the CLI silently
failed to load (Auth0 kept being served), costing a full deploy+debug cycle.

The merge now lets the CLI win over env: an env value is applied only where the
CLI did not explicitly set the arg; when the CLI did set it, the CLI value is
kept and the dropped env value is reported (so the caller can WARN).
"""
from hypha.server import get_argparser, merge_env_into_args


def _ns(*cli):
    """A parsed args namespace from the real argparser (no mocks)."""
    return get_argparser(add_help=False).parse_args(list(cli))


def test_cli_startup_functions_survives_from_env():
    """The headline bug: an explicit --startup-functions must survive --from-env."""
    defaults = _ns()
    cli = _ns("--startup-functions", "my.module:setup")
    env = _ns()  # environment carries startup_functions at its default (None)
    dropped = merge_env_into_args(cli, env, defaults)
    assert cli.startup_functions == ["my.module:setup"], (
        "explicit CLI --startup-functions must survive --from-env"
    )
    assert any(k == "startup_functions" for k, _, _ in dropped), (
        "the dropped env value for startup_functions must be reported (warned)"
    )


def test_cli_wins_over_conflicting_env_value():
    """When BOTH CLI and env set an arg, the CLI value wins."""
    defaults = _ns()
    cli = _ns("--startup-functions", "cli.module:setup")
    env = _ns("--startup-functions", "env.module:setup")
    dropped = merge_env_into_args(cli, env, defaults)
    assert cli.startup_functions == ["cli.module:setup"], "CLI must win over env"
    assert ("startup_functions", ["env.module:setup"], ["cli.module:setup"]) in dropped


def test_env_fills_unset_cli_args():
    """An env value IS applied when the CLI did not set that arg (the point of
    --from-env is still honored for args the operator didn't pass)."""
    defaults = _ns()
    cli = _ns()  # nothing set on the CLI
    env = _ns("--startup-functions", "env.module:setup")
    dropped = merge_env_into_args(cli, env, defaults)
    assert cli.startup_functions == ["env.module:setup"], (
        "env should fill args the CLI did not set"
    )
    assert dropped == [], "nothing should be dropped when the CLI set nothing"


def test_cli_scalar_survives_env_and_is_reported():
    """The precedence rule holds for scalar args too (e.g. --port)."""
    defaults = _ns()
    cli = _ns("--port", "1234")
    env = _ns("--port", "9999")
    dropped = merge_env_into_args(cli, env, defaults)
    assert cli.port == 1234, "explicit CLI --port must win over env"
    assert any(k == "port" for k, _, _ in dropped)


def test_env_scalar_applied_when_cli_at_default():
    """And env fills a scalar the CLI left at its default."""
    defaults = _ns()
    cli = _ns()
    env = _ns("--port", "9999")
    merge_env_into_args(cli, env, defaults)
    assert cli.port == 9999
