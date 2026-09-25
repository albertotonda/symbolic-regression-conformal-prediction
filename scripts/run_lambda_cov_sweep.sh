#!/usr/bin/env bash
# Launch one run_sigma_sr.py per lambda_cov value, each in its own tmux
# window of a shared session, so they run in parallel and stay individually
# attachable/killable. Optional first argument: base config name in
# src/configs/sigma-sr/ (default: default_config), e.g.
#     scripts/run_lambda_cov_sweep.sh lambda_sweep_base
set -euo pipefail

LAMBDA_VALUES=(500 1000 5000 10000 50000 100000)
SESSION="sigma_sr_sweep"
STAGGER_SECONDS=20 # spaces out PySR/Julia startup so they don't all hit the
                    # shared Julia depot's precompilation lock at once

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/src/configs/sigma-sr"
BASE_CONFIG="$CONFIG_DIR/${1:-default_config}.yaml"
LOG_DIR="$REPO_ROOT/logs/lambda_cov_sweep"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "Base config not found: $BASE_CONFIG" >&2
    exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' already exists. Attach with:" >&2
    echo "    tmux attach -t $SESSION" >&2
    echo "or kill it first with: tmux kill-session -t $SESSION" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

for i in "${!LAMBDA_VALUES[@]}"; do
    lambda="${LAMBDA_VALUES[$i]}"
    config_name="lambda_cov_${lambda}"
    config_path="$CONFIG_DIR/${config_name}.yaml"

    # per-lambda config: default_config.yaml with lambda_cov overridden
    sed -E "s/^lambda_cov[[:space:]]*:.*/lambda_cov : ${lambda}/" \
        "$BASE_CONFIG" > "$config_path"

    # `exec bash` keeps the window's pane open after the run finishes (or
    # fails) so output stays visible; without it, a fast failure on the
    # first window can kill the whole tmux server before later windows
    # are created.
    cmd="cd '$REPO_ROOT' && source .venv/bin/activate && sleep $((i * STAGGER_SECONDS)) && python src/run_sigma_sr.py -c ${config_name} 2>&1 | tee '$LOG_DIR/${config_name}.log'; exec bash"

    if [ "$i" -eq 0 ]; then
        tmux new-session -d -s "$SESSION" -n "lambda_${lambda}" "$cmd"
    else
        tmux new-window -t "$SESSION" -n "lambda_${lambda}" "$cmd"
    fi
done

echo "Launched ${#LAMBDA_VALUES[@]} runs in tmux session '$SESSION' (one window per lambda_cov value)."
echo "Attach with:   tmux attach -t $SESSION"
echo "List windows:  tmux list-windows -t $SESSION"
echo "Logs:          $LOG_DIR"
