#!/usr/bin/env bash
# Launch one T1 Self-Refine sweep: three seeds for one backbone on Tier 1 XSum. Thin wrapper:
# every experiment decision lives in the config and the package, none of it here.
#
#   ./src/shellscripts/run_t1_xsum.sh mistral-7b-instruct-v0.3 siddhanth
#
# Runs seeds 0, 1 and 2 in turn inside a tmux session named t1_<backbone>, so the sweep
# survives a dropped ssh connection. Attach with `tmux attach -t t1_<backbone>`.
#
# One backbone per invocation, and only one invocation at a time: vLLM wants the whole GPU and
# every run serves on the same endpoint, so concurrent sweeps fail the health check or OOM.
set -euo pipefail

backbone="${1:?usage: $0 <backbone> <owner> [extra --set overrides...]}"
owner="${2:?usage: $0 <backbone> <owner> [extra --set overrides...]}"
shift 2

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Empty when no overrides were passed: an unquoted "" would reach argparse as a stray
# empty argument and fail the run.
extra=""
if (($#)); then extra="$(printf "%q " "$@")"; fi
session="t1_${backbone%%-*}"

if tmux has-session -t "$session" 2>/dev/null; then
    echo "session $session already exists; attach with: tmux attach -t $session" >&2
    exit 1
fi

# `exec bash` keeps the window open on exit so a failed sweep's output stays readable.
tmux new-session -d -s "$session" -c "$repo" \
    "for s in 0 1 2; do \
        .venv/bin/python src/run_benchmark.py \
            --topology t1_self_refine \
            --backbone '$backbone' \
            --task tier1_xsum \
            --set run.owner='$owner' \
            --set run.seed=\$s \
            $extra; \
     done; exec bash"

echo "launched $session for $backbone (seeds 0 1 2)"
echo "attach:  tmux attach -t $session"
echo "detach:  ctrl-b then d"
