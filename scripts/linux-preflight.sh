#!/usr/bin/env bash
# Call with bash from the checkout root after activating the reviewed venv.
set -euo pipefail
umask 077
if (( $# != 3 )); then
    printf '%s\n' 'Usage: bash scripts/linux-preflight.sh COMMIT SCRATCH_PARENT NEW_MANIFEST' >&2
    exit 2
fi
if [[ $(uname -s) != Linux || ! -f experiments/deployment.py || -z ${VIRTUAL_ENV:-} ]]; then
    printf '%s\n' 'Requires Linux, the repository root, and an active virtual environment.' >&2
    exit 2
fi
if [[ ! $1 =~ ^[0-9a-f]{40}$ ]]; then
    printf '%s\n' 'Provide the full reviewed Git commit.' >&2
    exit 2
fi
# Thread values come from the verified allocation, not this script.
export MPLBACKEND=Agg PYTHONUTF8=1 PYTHONHASHSEED=0
exec "$VIRTUAL_ENV/bin/python" -B -m experiments.deployment preflight \
    --target-linux --include-tests --expected-commit "$1" \
    --scratch-parent "$2" --manifest "$3"
