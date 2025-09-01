set -euo pipefail
DSN="${MIGRATIONS_DSN:-postgresql://cortex:prodpassword@127.0.0.1:5440/cortex_rag}"
for f in database/migrations/*.sql; do
  echo ">> applying $f"
  psql "$DSN" -v ON_ERROR_STOP=1 -f "$f"
done
echo "OK"