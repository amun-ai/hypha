# Login and Token Acquisition

This guide explains how clients obtain authentication tokens for a Hypha server.
Hypha tokens authorize all RPC and HTTP calls — once you have a token, pass it via
`connect_to_server({..., "token": token})` or the `Authorization: Bearer <token>` HTTP
header.

There are three ways to get a token:

1. **Interactive browser login** — `login()` opens an OAuth flow (Auth0 by default).
2. **Programmatic token generation** — `server.generate_token(...)` after you're already connected as an admin.
3. **Pre-shared token** — paste a long-lived token into configuration (e.g. `HYPHA_TOKEN` env var).

Most AI agent deployments use option 2 or 3. Option 1 is for human users and for
CLI tools that need to bootstrap a token on a new machine.

## Interactive Browser Login — Python

`hypha_rpc.login(config)` connects to the `public/hypha-login` service, starts an
OAuth session, prints a URL for the user to open, then polls until the user
completes the flow and returns a token.

```python
from hypha_rpc import login, connect_to_server

# Default flow: prints the URL to stdout and waits
token = await login({
    "server_url": "https://hypha.aicell.io",
    # Optional:
    # "workspace": "ws-user-xyz",     # request a token scoped to a workspace
    # "expires_in": 3600 * 24 * 30,   # seconds; default is server-configured
    # "login_timeout": 180,           # how long to wait for user to finish
    # "profile": True,                # return {"token": ..., "user": {...}}
})

async with connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": token,
}) as server:
    print(await server.check_status())
```

### Custom `login_callback`

By default `login()` prints the login URL. In a GUI app, notebook, bot, or
headless environment, provide a `login_callback` to display the URL however you
like. The callback receives a dict with `login_url`, `key`, and `report_url`:

```python
async def my_callback(context):
    # context = {"login_url": "https://...", "key": "...", "report_url": "..."}
    print("Open this link to sign in:", context["login_url"])
    # e.g. send to Slack, render a QR code, open a webview, etc.
    # await bot.send_message(chat_id, context["login_url"])

token = await login({
    "server_url": "https://hypha.aicell.io",
    "login_callback": my_callback,
    "login_timeout": 300,
})
```

The callback may be `async` or sync. `login()` returns only after the user
finishes the browser flow or the timeout elapses.

### Logout

```python
from hypha_rpc import logout

await logout({"server_url": "https://hypha.aicell.io"})
# Opens a logout URL; pass logout_callback=... to customize.
```

## Interactive Browser Login — JavaScript

```javascript
import { login, connectToServer } from "hypha-rpc";

const token = await login({
  server_url: "https://hypha.aicell.io",
  // Optional:
  // workspace: "ws-user-xyz",
  // expires_in: 3600 * 24 * 30,
  // login_timeout: 180,
  // profile: true,
  login_callback: async (context) => {
    // context.login_url — show it to the user
    window.open(context.login_url, "_blank");
  },
});

const server = await connectToServer({
  server_url: "https://hypha.aicell.io",
  token,
});
```

Without `login_callback`, `login()` logs the URL via `console.log`. In browser
apps, always provide a callback that opens a popup or redirects — users can't
see `console` output.

### Logout

```javascript
import { logout } from "hypha-rpc";
await logout({ server_url: "https://hypha.aicell.io" });
```

## Programmatic Tokens — `generate_token`

Once connected with any authenticated identity, you can mint scoped tokens via
the workspace manager (requires `admin` permission on the target workspace):

```python
async with connect_to_server({
    "server_url": "https://hypha.aicell.io",
    "token": admin_token,
}) as server:
    new_token = await server.generate_token({
        "workspace": "ws-user-xyz",        # workspace to scope to
        "permission": "read_write",         # read | read_write | admin
        "expires_in": 86400,                # seconds
        # Optional: "client_id", "user_id", "email"
    })
```

Use this to hand out short-lived tokens to sub-agents, CI jobs, or users of a
hosted app. The token inherits permissions no greater than the caller's own.

## Token Expiration and Refresh

Tokens carry a `exp` claim. On expiration, RPC calls fail with
`ConnectionAbortedError: Authentication error: The token has expired`. Handle
this by calling `login()` again or refreshing via `generate_token` from a
longer-lived admin session.

`hypha-rpc` does **not** auto-refresh tokens. If you need long-running agents,
either:

- Issue tokens with a long `expires_in` (at the cost of revocability), or
- Wrap your connection in a reconnection loop that calls `login()` / `generate_token()` on auth failure.

## Custom Auth Providers

The `public/hypha-login` service and `login()` function assume the server uses
the built-in Auth0 integration. If your Hypha deployment uses a custom auth
provider (see [auth.md](auth.md)), use the provider-specific token flow and
pass the resulting token directly to `connect_to_server`. You do not need
`login()` at all — it's purely a convenience wrapper around
`public/hypha-login`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to get the login service: public/hypha-login` | Server has no login service configured | Use `generate_token` or a pre-shared token instead |
| `login_timeout` exceeded | User didn't finish the flow in time | Pass a larger `login_timeout` or a UI-friendly `login_callback` |
| `The token has expired` | Token's `exp` passed | Re-run `login()` or `generate_token(...)` |
| `PermissionError` on `generate_token` | Caller is not `admin` on target workspace | Use an admin token or request a smaller scope |
| Browser callback never returns | Popup blocked, or wrong `report_url` domain | Allow popups, or display `login_url` in a clickable link |

## See Also

- [auth.md](auth.md) — Server-side authentication, Auth0 config, custom providers
- [configurations.md](configurations.md) — Server startup flags and environment variables
- `public/hypha-login` service reference — underlying RPC surface that `login()` wraps
