#!/usr/bin/env bash
# Draw and record a frozen evaluation sample. Thin wrapper: every decision lives in the
# package module and in the config, none of it here.
#
#   ./src/shellscripts/freeze_eval_sample.sh src/configs/data/tier1_xsum.yaml sameer
#
# Running this a second time refuses rather than redrawing, because redrawing changes which
# documents every topology is evaluated on. See decision D-006.
set -euo pipefail

config="${1:?usage: $0 <config.yaml> <owner> [extra --set overrides...]}"
owner="${2:?usage: $0 <config.yaml> <owner> [extra --set overrides...]}"
shift 2

out="${config%.yaml}_sample.yaml"

python -m agent_evals.datasets.freeze \
    --config "$config" \
    --out "$out" \
    --set "run.owner=$owner" \
    "$@"
