#!/usr/bin/env bash
# Prompt for TNS bot credentials and write them into the repo's .env.
# The API key is read without echo; no value is ever printed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$ROOT/.env"

read -rsp "TNS_API_KEY (input hidden): " api_key; echo
read -rp  "TNS_TNS_ID (numeric bot ID): " tns_id
read -rp  "TNS_MARKER_NAME (bot name): " marker_name
read -rp  "TNS_MARKER_TYPE [bot]: " marker_type
marker_type="${marker_type:-bot}"

if [[ -z "$api_key" || -z "$tns_id" || -z "$marker_name" ]]; then
  echo "API key, TNS ID and marker name are all required; .env left unchanged." >&2
  exit 1
fi

touch "$ENV_FILE"
tmp="$(mktemp "$ENV_FILE.XXXXXX")"
trap 'rm -f "$tmp"' EXIT
chmod 600 "$tmp"

# Keep every non-TNS line, then append the new TNS block.
grep -vE '^(TNS_API_KEY|TNS_TNS_ID|TNS_MARKER_NAME|TNS_MARKER_TYPE)=' "$ENV_FILE" > "$tmp" || true
{
  printf 'TNS_API_KEY=%s\n'     "$api_key"
  printf 'TNS_TNS_ID=%s\n'      "$tns_id"
  printf 'TNS_MARKER_NAME=%s\n' "$marker_name"
  printf 'TNS_MARKER_TYPE=%s\n' "$marker_type"
} >> "$tmp"
mv "$tmp" "$ENV_FILE"
trap - EXIT
unset api_key

echo "Updated TNS_* in $ENV_FILE (mode 600)."

# Check that the pipeline can load them (repr redacts the key).
PY="${HOME}/.venvs/debass_py313/bin/python"
[[ -x "$PY" ]] || PY=python3
(cd "$ROOT" && "$PY" -c '
import sys; sys.path.insert(0, "src")
from dotenv import load_dotenv; load_dotenv(".env", override=True)
from debass_meta.access.tns import load_tns_credentials
print("Loaded:", load_tns_credentials())
')
