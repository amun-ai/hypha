# Hypha CLI — Command-Line Interface for Agents

> **Install:** `npm install -g hypha-cli` (requires Node.js >= 22)
>
> The `hypha` CLI lets code agents interact with Hypha servers directly from the terminal — no SDK code needed. This is ideal for AI coding agents that can run shell commands.

## Setup

```bash
# Install globally
npm install -g hypha-cli

# Login (opens browser for OAuth)
hypha login https://hypha.aicell.io

# Or set credentials directly
export HYPHA_SERVER_URL=https://hypha.aicell.io
export HYPHA_TOKEN=your-token-here
export HYPHA_WORKSPACE=your-workspace
```

Config is stored in `~/.hypha/.env`. Environment variables override config file values.

## Command Reference

### Workspace Commands

```bash
hypha login [server-url]                                    # OAuth login via browser
hypha token [--expires-in N] [--permission P] [--workspace W] [--json]  # Generate token
hypha services [--type T] [--include-unlisted] [--json]     # List services
hypha info [--json]                                         # Workspace info
```

### App Lifecycle Commands

```bash
hypha apps list [--json]                                    # List installed app definitions
hypha apps info <app-id> [--json]                           # Show app details
hypha apps install <source> [--id ID] [--overwrite]         # Install app from URL or source
hypha apps uninstall <app-id>                               # Remove app definition (alias: rm)
hypha apps start <app-id> [--timeout N] [--wait]            # Start app instance
hypha apps stop <id> [--all]                                # Stop instance(s)
hypha apps ps [--json]                                      # List running instances
hypha apps logs <instance-id> [--tail N] [--type T]         # View instance logs
```

### Artifact Commands

Artifact addressing: `alias` or `workspace/alias`. File paths: `artifact:path/to/file`.
Shorthand: `hypha art` = `hypha artifacts`.

```bash
hypha artifacts ls [artifact[:path]] [--json] [--long]      # List artifacts or files
hypha artifacts cat <artifact:path>                          # Print file to stdout
hypha artifacts cp <src> <dest> [-r] [--commit]             # Upload/download (scp-style)
hypha artifacts rm <artifact[:path]> [-r] [--force]         # Delete artifact or files
hypha artifacts create <alias> [--type T] [--parent P]      # Create artifact (alias: mkdir)
hypha artifacts info <artifact> [--json]                     # Artifact metadata
hypha artifacts search <query> [--type T] [--limit N]       # Search (alias: find)
hypha artifacts commit <artifact> [--version V] [--message M]  # Commit staged changes
hypha artifacts edit <artifact> [--name N] [--description D]   # Edit metadata
hypha artifacts discard <artifact>                           # Discard staged changes
```

### Global Options

```bash
--server <url>       Override Hypha server URL
--workspace <ws>     Override workspace ID
--help, -h           Show help for any command
--version, -v        Show CLI version
```

## Common Agent Workflows

### 1. Browse and Inspect Services

```bash
# List all services in current workspace
hypha services --json

# List services in a specific workspace
hypha services --workspace my-workspace --json
```

### 2. Manage Server Apps

```bash
# Install an app from a URL
hypha apps install https://example.com/my-app.py --id my-app

# Start the app and wait for it to be ready
hypha apps start my-app --wait --timeout 30

# Check running instances
hypha apps ps --json

# View logs
hypha apps logs <instance-id> --tail 50

# Stop the app
hypha apps stop my-app --all
```

### 3. Work with Artifacts (Files & Data)

```bash
# List all artifacts
hypha art ls --json

# Create a new artifact
hypha art create my-dataset --type dataset

# Upload files
hypha art cp ./data.csv my-dataset:data/train.csv
hypha art cp ./dataset/ my-dataset:data/ -r       # Recursive upload

# Upload and auto-commit in one step
hypha art cp ./results.json my-dataset:results.json --commit

# Download files
hypha art cp my-dataset:data/train.csv ./local/

# Browse files in an artifact
hypha art ls my-dataset:data/ --long

# View file contents
hypha art cat my-dataset:data/config.yaml

# Commit staged changes
hypha art commit my-dataset --version 1.0.0 --message "Initial dataset"

# Search artifacts
hypha art search "training data" --type dataset --limit 10

# Delete files
hypha art rm my-dataset:data/old-file.csv --force
```

### 4. Token Management

```bash
# Generate a read-write token (requires admin permission)
hypha token --permission read_write --expires-in 86400 --json

# Generate a read-only token for a specific workspace
hypha token --permission read --workspace shared-data --json
```

## Architecture

| Command group | Transport | Service |
|---|---|---|
| Workspace (login, token, services, info) | RPC (hypha-rpc) | Server methods directly |
| Apps | RPC (hypha-rpc) | `public/server-apps` service |
| Artifacts metadata (create, edit, commit, delete, search) | RPC (hypha-rpc) | `public/artifact-manager` service |
| Artifacts files (ls files, cat, cp) | HTTP with Bearer token | REST endpoints |

File transfers use presigned URLs: upload via `put_file()` RPC then HTTP PUT, download via `get_file()` RPC then HTTP GET.

## Key Concepts

- **Artifact staging**: Files are staged before committing. Use `--commit` flag with `cp` for auto-commit, or manually call `commit` after uploading.
- **Presigned URLs**: File uploads/downloads go through S3-compatible presigned URLs, not through RPC.
- **Artifact types**: `generic`, `collection`, `application`, `model`, `dataset`, `skill`, etc.
- **Cross-workspace access**: Use `workspace/alias` format to reference artifacts in other workspaces.
- **JSON output**: Most commands support `--json` flag for machine-readable output, ideal for agent parsing.

## Tips for AI Agents

1. **Always use `--json`** when you need to parse command output programmatically
2. **Use `hypha art` shorthand** instead of `hypha artifacts` to save tokens
3. **Prefer `cp --commit`** to combine upload and commit in a single command
4. **Check `hypha info --json`** first to verify connectivity and permissions
5. **Use `hypha apps ps --json`** to get instance IDs before viewing logs or stopping apps
6. **Environment variables** (`HYPHA_SERVER_URL`, `HYPHA_TOKEN`, `HYPHA_WORKSPACE`) avoid repeating `--server` and `--workspace` flags
