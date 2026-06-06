#!/usr/bin/env bash
#
# Daily logical backup of the predictor's PostgreSQL database (NFR-06).
#
# Produces a compressed custom-format dump (pg_dump -Fc, restorable via
# pg_restore) and keeps only the newest N snapshots. Designed to run from a
# cron entry on the host/volume or from the scheduled GitHub Actions workflow
# (.github/workflows/backup.yml).
#
# Configuration via environment:
#   DATABASE_URL        full connection string (wins if set; e.g. Fly Postgres)
#   POSTGRES_*          host/port/user/password/db parts (fallback)
#   BACKUP_DIR          output directory (default: ./backups)
#   BACKUP_RETENTION    how many dumps to keep (default: 30)
#
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-backups}"
RETENTION="${BACKUP_RETENTION:-30}"

DB_URL="${DATABASE_URL:-postgresql://${POSTGRES_USER:-cicd_predictor}:${POSTGRES_PASSWORD:-changeme}@${POSTGRES_HOST:-localhost}:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-cicd_predictor}}"

mkdir -p "$BACKUP_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$BACKUP_DIR/cicd_predictor_${TS}.dump"

echo "[backup] dumping database -> $OUT"
pg_dump -Fc --no-owner --no-privileges "$DB_URL" -f "$OUT"
echo "[backup] wrote $(du -h "$OUT" | cut -f1) to $OUT"

# Rotation: keep the newest $RETENTION dumps, delete the rest. Portable across
# bash 3.2 (macOS) and bash 5 (CI) — no mapfile / array-slice features.
ls -1t "$BACKUP_DIR"/cicd_predictor_*.dump 2>/dev/null | tail -n +"$((RETENTION + 1))" | while IFS= read -r old; do
  echo "[backup] rotate: removing old snapshot $old"
  rm -f "$old"
done

KEPT="$(ls -1 "$BACKUP_DIR"/cicd_predictor_*.dump 2>/dev/null | wc -l | tr -d ' ')"
echo "[backup] done — $KEPT snapshot(s) retained (limit $RETENTION)."
