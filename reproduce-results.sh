#!/bin/bash

generate_and_judge() {
    api=$1
    model=$2
    concurrency=$3

    echo "Generating $api/$model with concurrency $concurrency"
    python research-eval.py generate --api $api --model $model -o experiments/$api.$model.out.jsonl --concurrency $concurrency
    echo "Judging $api/$model"
    python research-eval.py judge -i experiments/$api.$model.out.jsonl -o experiments/$api.$model.eval.jsonl
    echo
}

mkdir -p experiments

generate_and_judge reka reka-flash-research-20250709 20
generate_and_judge google gemini-2.0-flash 5
generate_and_judge openai gpt-4o-search-preview-2025-03-11 30
generate_and_judge anthropic claude-sonnet-4-20250514 3
generate_and_judge perplexity sonar-reasoning-pro 30
generate_and_judge mistral mistral-medium-2505 3
generate_and_judge xai grok-4-0709 10

echo "Results:"
for f in experiments/*.eval.jsonl; do
    python research-eval.py report -i $f
done
