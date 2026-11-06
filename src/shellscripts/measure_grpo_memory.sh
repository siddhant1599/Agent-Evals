#!/usr/bin/env bash
# Run the GRPO memory measurement (Q-004, D-055): every cell of the grid, one training process
# each. Thin wrapper: the grid lives in src/configs/rl/grpo_xsum_memory.yaml and every decision
# in agent_evals.rl.memory, none of it here.
#
#   ./src/shellscripts/measure_grpo_memory.sh sameer           # the grid: longest articles
#   ./src/shellscripts/measure_grpo_memory.sh sameer typical   # timing: median-length articles
#   ./src/shellscripts/measure_grpo_memory.sh sameer forced    # longest articles, full summaries
#
# Run on the GPU instance from the project venv, with nothing else on the GPU: a second process
# would inflate every cell's peak VRAM. Each cell writes its own results/<run_id>/.
set -euo pipefail

owner="${1:?usage: $0 <owner> [typical|forced]}"
case "${2:-}" in
    "") step=sweep ;;
    typical) step=sweep-typical ;;
    forced) step=sweep-forced ;;
    *) echo "usage: $0 <owner> [typical|forced]" >&2; exit 1 ;;
esac
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found: run this on the GPU instance" >&2
    exit 1
fi
used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)"
if [ "$used_mib" -gt 0 ]; then
    echo "the GPU already holds ${used_mib} MiB; stop other GPU jobs first" >&2
    exit 1
fi

python -m agent_evals.rl.memory "$step" \
    --config src/configs/rl/grpo_xsum_memory.yaml \
    --set "run.owner=$owner"
