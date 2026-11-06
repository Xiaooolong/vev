#!/usr/bin/env bash
# Start `vev serve` with the reading layout of the Decision Index runs (docs/serving.md): bidirectional state block,
# text states up to 3,000 tokens and images read twice, images up to 1.6 MP. The warm-up runs before the server
# answers /healthz. Arguments after MODEL go to `vev serve` (for example --port, --host, --revision).
#
#   scripts/serve_board.sh CountingSheep/vev-4b-v2 --port 8009
set -euo pipefail
if [ $# -lt 1 ]; then
  echo "usage: $0 MODEL [vev serve options]" >&2
  exit 2
fi
model=$1
shift
export VEV_BIDIR=state VEV_READ_TWICE=full VEV_READ_TWICE_MAX=3000 VEV_IMAGE_READ=twice VEV_MAX_IMAGE_PIXELS=1638400
exec "${PYTHON:-python}" -m vev serve --model "$model" "$@"
