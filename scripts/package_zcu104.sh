#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$PROJECT_DIR/deploy/zcu104"
MODEL_SOURCE="$PROJECT_DIR/models/compiled/zcu104/trafficsignnet_int8.xmodel"
MODEL_TARGET="$DEPLOY_DIR/model/trafficsignnet_int8.xmodel"
ARCHIVE="$PROJECT_DIR/artifacts/zcu104/trafficsignnet_zcu104.tar.gz"

if [[ ! -f "$MODEL_SOURCE" ]]; then
  echo "ERROR: canonical XMODEL not found: $MODEL_SOURCE" >&2
  exit 1
fi

mkdir -p "$DEPLOY_DIR/model" "$(dirname "$ARCHIVE")"
cp "$MODEL_SOURCE" "$MODEL_TARGET"
tar --exclude="__pycache__" --exclude="build" -C "$PROJECT_DIR/deploy" -czf "$ARCHIVE" zcu104
sha256sum "$ARCHIVE"
echo "Created: $ARCHIVE"
