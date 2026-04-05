#!/usr/bin/env bash
# run_plan.sh — One-shot planning: analyze → recommend → export
# Usage: ./scripts/run_plan.sh <benchmark_dir> [--sla-ttft 200] [--sla-tpot 50]
set -euo pipefail

BENCHMARK_DIR="${1:?Usage: $0 <benchmark_dir> [--sla-ttft MS] [--sla-tpot MS]}"
shift

SLA_TTFT=200
SLA_TPOT=50
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sla-ttft) SLA_TTFT="$2"; shift 2 ;;
    --sla-tpot) SLA_TPOT="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -d "$BENCHMARK_DIR" ]]; then
  echo "Error: $BENCHMARK_DIR is not a directory" >&2
  exit 1
fi

BENCHMARKS=$(find "$BENCHMARK_DIR" -name '*.json' -type f | sort)
if [[ -z "$BENCHMARKS" ]]; then
  echo "Error: no .json files found in $BENCHMARK_DIR" >&2
  exit 1
fi

BENCH_ARGS=()
while IFS= read -r f; do
  BENCH_ARGS+=(--benchmark "$f")
done <<< "$BENCHMARKS"

SLA_ARGS=(--sla-ttft "$SLA_TTFT" --sla-tpot "$SLA_TPOT")

echo "=== Step 1/3: Analyze ==="
xpyd-plan analyze "${BENCH_ARGS[@]}" "${SLA_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo ""
echo "=== Step 2/3: Recommend ==="
xpyd-plan recommend "${BENCH_ARGS[@]}" "${SLA_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo ""
echo "=== Step 3/3: Export ==="
OUTPUT_DIR="${BENCHMARK_DIR}/plan-output"
mkdir -p "$OUTPUT_DIR"
xpyd-plan export "${BENCH_ARGS[@]}" "${SLA_ARGS[@]}" \
  --format json --output "$OUTPUT_DIR/results.json" "${EXTRA_ARGS[@]}"
xpyd-plan export "${BENCH_ARGS[@]}" "${SLA_ARGS[@]}" \
  --format csv --output "$OUTPUT_DIR/results.csv" "${EXTRA_ARGS[@]}"

echo ""
echo "Done. Results exported to $OUTPUT_DIR/"
