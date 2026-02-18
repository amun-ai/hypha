#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="$ROOT_DIR/docker-compose"
COMPOSE_BASE="$COMPOSE_DIR/docker-compose.yml"
COMPOSE_OVERRIDE="$COMPOSE_DIR/docker-compose.live-e2e.yml"
ENV_TEMPLATE="$COMPOSE_DIR/.env-docker"
ENV_FILE="$COMPOSE_DIR/.env.live-e2e"

PROJECT_NAME="${HYPHA_LIVE_E2E_PROJECT:-hypha-live-e2e}"
HYPHA_MODE="${HYPHA_LIVE_E2E_HYPHA_MODE:-host}"
SERVER_HOST="${HYPHA_LIVE_E2E_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${HYPHA_LIVE_E2E_SERVER_PORT:-9527}"
SERVER_URL="${HYPHA_LIVE_E2E_SERVER_URL:-http://${SERVER_HOST}:${SERVER_PORT}}"
ROOT_TOKEN="${HYPHA_LIVE_E2E_ROOT_TOKEN:-09zDo-WV2_ZLwVfTA9Gj-pGKs2X403nio-StS2e-JihUBAiPW3hXsQ}"
WEBHOOK_SECRET="${HYPHA_LIVE_E2E_WEBHOOK_SECRET:-whsec_contract_fixture_secret}"
KEEP_STACK="${HYPHA_LIVE_E2E_KEEP_STACK:-0}"
WAIT_TIMEOUT_SECONDS="${HYPHA_LIVE_E2E_WAIT_TIMEOUT_SECONDS:-300}"
SERVER_LOG_FILE="${HYPHA_LIVE_E2E_SERVER_LOG_FILE:-$ROOT_DIR/.codex/logs/live-e2e-hypha.log}"
HYPHA_PYTHON_BIN="${HYPHA_LIVE_E2E_PYTHON:-$ROOT_DIR/.venv/bin/python}"

if [[ ! -f "$COMPOSE_BASE" ]]; then
  echo "Missing compose file: $COMPOSE_BASE" >&2
  exit 1
fi
if [[ ! -f "$ENV_TEMPLATE" ]]; then
  echo "Missing env template: $ENV_TEMPLATE" >&2
  exit 1
fi
if [[ "$HYPHA_MODE" == "compose" && ! -f "$COMPOSE_OVERRIDE" ]]; then
  echo "Missing compose override file: $COMPOSE_OVERRIDE" >&2
  exit 1
fi

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD=(docker-compose)
else
  echo "Docker Compose is not available. Install docker compose or docker-compose." >&2
  exit 1
fi

COMPOSE_ARGS=(--env-file "$ENV_FILE" -p "$PROJECT_NAME" -f "$COMPOSE_BASE")
if [[ "$HYPHA_MODE" == "compose" ]]; then
  COMPOSE_ARGS+=(-f "$COMPOSE_OVERRIDE")
fi

compose() {
  "${COMPOSE_CMD[@]}" "${COMPOSE_ARGS[@]}" "$@"
}

wait_for_service_health() {
  local service="$1"
  local timeout_seconds="$2"
  local start_ts now_ts container_id status

  start_ts="$(date +%s)"
  container_id="$(compose ps -q "$service" | head -n 1)"
  if [[ -z "$container_id" ]]; then
    echo "Could not find container id for service '$service'." >&2
    return 1
  fi

  while true; do
    status="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id" 2>/dev/null || true)"
    if [[ "$status" == "healthy" || "$status" == "running" ]]; then
      return 0
    fi
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= timeout_seconds )); then
      echo "Service '$service' did not become ready in ${timeout_seconds}s (status=${status:-unknown})." >&2
      return 1
    fi
    sleep 2
  done
}

mkdir -p \
  "$COMPOSE_DIR/data/live-e2e/hypha" \
  "$COMPOSE_DIR/data/live-e2e/postgres" \
  "$COMPOSE_DIR/data/live-e2e/redis" \
  "$COMPOSE_DIR/data/live-e2e/minio"
mkdir -p "$(dirname "$SERVER_LOG_FILE")"

cp "$ENV_TEMPLATE" "$ENV_FILE"
cat >>"$ENV_FILE" <<EOF
HYPHA_ROOT_TOKEN=$ROOT_TOKEN
HYPHA_STRIPE_WEBHOOK_SECRET=$WEBHOOK_SECRET
HYPHA_DATA_DIR=./data/live-e2e/hypha
POSTGRES_DATA_DIR=./data/live-e2e/postgres
REDIS_DATA_DIR=./data/live-e2e/redis
MINIO_DATA_DIR=./data/live-e2e/minio
HYPHA_RESET_REDIS=true
EOF

if [[ ! -x "$HYPHA_PYTHON_BIN" ]]; then
  HYPHA_PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "$HYPHA_PYTHON_BIN" ]]; then
  echo "Could not find Python executable for host Hypha mode." >&2
  exit 1
fi

POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-mysecretpassword}"
POSTGRES_DB="${POSTGRES_DB:-hypha_db}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
REDIS_PASSWORD="${REDIS_PASSWORD:-redis123}"
REDIS_PORT="${REDIS_PORT:-6379}"
DATABASE_URI="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT}/${POSTGRES_DB}"
REDIS_URI="redis://:${REDIS_PASSWORD}@127.0.0.1:${REDIS_PORT}/0"

HYPHA_PID=""
cleanup() {
  if [[ -n "$HYPHA_PID" ]] && kill -0 "$HYPHA_PID" >/dev/null 2>&1; then
    kill "$HYPHA_PID" >/dev/null 2>&1 || true
    wait "$HYPHA_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$KEEP_STACK" == "1" ]]; then
    echo "Keeping stack running (HYPHA_LIVE_E2E_KEEP_STACK=1)."
    return
  fi
  compose down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Starting live E2E dependencies with project '$PROJECT_NAME'..."
compose up -d --remove-orphans postgres redis
wait_for_service_health postgres "$WAIT_TIMEOUT_SECONDS"
wait_for_service_health redis "$WAIT_TIMEOUT_SECONDS"
if [[ "$HYPHA_MODE" == "compose" ]]; then
  echo "Starting Hypha in compose mode..."
  compose up -d --remove-orphans hypha-server
else
  echo "Starting Hypha in host mode using $HYPHA_PYTHON_BIN ..."
  HYPHA_ROOT_TOKEN="$ROOT_TOKEN" \
  HYPHA_STRIPE_WEBHOOK_SECRET="$WEBHOOK_SECRET" \
  HYPHA_LOGLEVEL="${HYPHA_LOGLEVEL:-INFO}" \
  "$HYPHA_PYTHON_BIN" -m hypha.server \
    --host="$SERVER_HOST" \
    --port="$SERVER_PORT" \
    --database-uri="$DATABASE_URI" \
    --redis-uri="$REDIS_URI" \
    --migrate-database=head \
    --reset-redis \
    --root-token="$ROOT_TOKEN" \
    >"$SERVER_LOG_FILE" 2>&1 &
  HYPHA_PID="$!"
fi

echo "Waiting for readiness at $SERVER_URL/health/readiness ..."
start_ts="$(date +%s)"
until curl -fsS "$SERVER_URL/health/readiness" >/dev/null 2>&1; do
  now_ts="$(date +%s)"
  if (( now_ts - start_ts >= WAIT_TIMEOUT_SECONDS )); then
    echo "Server did not become ready in ${WAIT_TIMEOUT_SECONDS}s." >&2
    compose ps >&2 || true
    if [[ "$HYPHA_MODE" == "compose" ]]; then
      compose logs --tail=200 hypha-server >&2 || true
    else
      tail -n 200 "$SERVER_LOG_FILE" >&2 || true
    fi
    exit 1
  fi
  sleep 2
done

if [[ -x "$ROOT_DIR/.venv/bin/pytest" ]]; then
  PYTEST_BIN="$ROOT_DIR/.venv/bin/pytest"
else
  PYTEST_BIN="pytest"
fi

TEST_TARGETS=("tests/live/test_logging_interception_billing_live_e2e.py")
if (( $# > 0 )); then
  TEST_TARGETS=("$@")
fi

echo "Running Phase 6 live E2E tests..."
HYPHA_LIVE_E2E=1 \
HYPHA_LIVE_E2E_SERVER_URL="$SERVER_URL" \
HYPHA_LIVE_E2E_ROOT_TOKEN="$ROOT_TOKEN" \
HYPHA_LIVE_E2E_WEBHOOK_SECRET="$WEBHOOK_SECRET" \
"$PYTEST_BIN" -v "${TEST_TARGETS[@]}"

echo "Phase 6 live E2E completed successfully."
