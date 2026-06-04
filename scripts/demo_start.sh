#!/usr/bin/env bash
# Start the full demo stack: postgres, redis, backend, frontend, cloudflared tunnel.
# Updates the GitHub webhook URL to the freshly minted tunnel and enables Actions
# on the demo repo. Idempotent — safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="/tmp/cicd-demo"
mkdir -p "$LOGDIR"

cd "$ROOT"

# Load env so backend + scripts use the right secrets.
set -a
. ./.env
set +a

# 1. Infrastructure containers.
echo "▶ Postgres + Redis (docker compose)..."
docker compose up -d postgres redis >/dev/null

# 2. Backend (FastAPI).
echo "▶ Backend on :8000..."
pkill -f "uvicorn app.main" 2>/dev/null || true
sleep 1
(
  cd backend
  nohup uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 \
    >"$LOGDIR/backend.log" 2>&1 &
)
sleep 4
curl -sf http://127.0.0.1:8000/health >/dev/null && echo "  ✓ backend healthy"

# 3. Frontend (Vite).
echo "▶ Frontend on :3000..."
pkill -f "vite --port 3000" 2>/dev/null || true
sleep 1
(
  cd frontend
  nohup npm run dev >"$LOGDIR/frontend.log" 2>&1 &
)
sleep 4
echo "  ✓ frontend booting → http://localhost:3000"

# 4. Cloudflared quick tunnel.
echo "▶ Cloudflared tunnel..."
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1
nohup cloudflared tunnel --url http://localhost:8000 --no-autoupdate \
  >"$LOGDIR/cloudflared.log" 2>&1 &

# Poll the tunnel URL out of cloudflared log.
TUNNEL=""
for _ in $(seq 1 30); do
  sleep 1
  TUNNEL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOGDIR/cloudflared.log" | head -1 || true)
  [ -n "$TUNNEL" ] && break
done
if [ -z "$TUNNEL" ]; then
  echo "  ✗ tunnel URL not found in log; aborting"
  exit 1
fi
echo "  ✓ tunnel: $TUNNEL"

# 5. Patch demo repo webhook to the new tunnel URL.
WEBHOOK_URL="$TUNNEL/api/v1/webhook/github"
echo "▶ Updating webhook on Spedymax/cicd-predictor-demo..."
gh api -X PATCH repos/Spedymax/cicd-predictor-demo/hooks/623494949 \
  -f "config[url]=$WEBHOOK_URL" \
  -f "config[content_type]=json" \
  -f "config[secret]=$GITHUB_WEBHOOK_SECRET" \
  -f "config[insecure_ssl]=0" >/dev/null
echo "  ✓ webhook → $WEBHOOK_URL"

# 6. Make sure Actions are enabled on the demo repo.
gh api -X PUT repos/Spedymax/cicd-predictor-demo/actions/permissions \
  -F enabled=true -f allowed_actions=all >/dev/null
echo "  ✓ Actions enabled"

echo ""
echo "Stack is ready. Open the dashboard:  http://localhost:3000"
echo "Tunnel URL:                          $TUNNEL"
echo "Logs:                                $LOGDIR/"
echo ""
echo "Next: ./scripts/demo_run.sh"
