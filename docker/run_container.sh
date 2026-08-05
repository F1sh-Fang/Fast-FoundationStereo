#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTAINER_NAME="${FFS_CONTAINER_NAME:-ffs}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  exec "${SCRIPT_DIR}/2_start_container.sh"
else
  exec "${SCRIPT_DIR}/1_create_container.sh"
fi
