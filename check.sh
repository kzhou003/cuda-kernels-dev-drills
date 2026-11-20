#!/usr/bin/env bash
# Run the acceptance tests for one drill (or all of them).
#   ./check.sh a1                 # test your mine/a1_*.py
#   ./check.sh a1 -x              # extra args go to pytest
#   ./check.sh all                # every drill (G drills are skipped without a CUDA GPU)
#   ./check.sh a1 --solutions     # run the reference solution instead (DRILL_IMPL=solutions)
#   ./check.sh g2 --bench         # G track only: benchmark against cuBLAS / SDPA (needs the GPU)
# Python: $PYTHON if set, else .venv/bin/python, else python3. G drills need torch with CUDA + nvcc.
set -euo pipefail
cd "$(dirname "$0")"

id="${1:-all}"
shift || true
args=()
bench=0
for a in "$@"; do
  case "$a" in
    --solutions) export DRILL_IMPL=solutions ;;
    --bench) bench=1 ;;
    *) args+=("$a") ;;
  esac
done

if [[ -n "${PYTHON:-}" ]]; then
  py="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  py=.venv/bin/python
else
  py=python3
fi
if ! "$py" -c "import pytest" 2>/dev/null; then
  echo "first run: creating .venv with pytest + numpy (for the G track use a Python that has torch+CUDA: PYTHON=...)" >&2
  python3 -m venv .venv && .venv/bin/pip install -q pytest numpy
  py=.venv/bin/python
fi

if (( bench )); then
  exec "$py" bench.py "$id"
fi

if [[ "$id" == "all" ]]; then
  targets=(tests)
else
  shopt -s nullglob
  targets=(tests/test_"${id}"_*.py)
  if (( ${#targets[@]} == 0 )); then
    echo "no test file matches tests/test_${id}_*.py" >&2
    exit 1
  fi
fi
exec "$py" -m pytest -q "${targets[@]}" ${args[@]+"${args[@]}"}
