#!/usr/bin/env sh

# Project Axiom: Young Link Matchup-Specific Training
#
# Trains specialized models for key matchups:
# - Priority 1: Jigglypuff, Peach (favorable matchups to maximize)
# - Priority 2: Fox, Falco (common opponents)
# - Priority 3: Falcon, Marth (difficult matchups to minimize losses)
#
# Each matchup gets 50k training steps of focused self-play.

set -e

CHAR=younglink
DELAY=18
BASE_TAG="axiom_yl_matchup"

# Paths
DOLPHIN_PATH="${DOLPHIN_PATH:-/Applications/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin}"
ISO_PATH="${ISO_PATH:-./SSBM.iso}"
OPTIMIZED_PATH="${MODEL_ROOT:-./models}/rl_optimized/axiom_yl_rl_optimization_best.pkl"
OUTPUT_DIR="${MODEL_ROOT:-./models}/matchups"

echo "=== Project Axiom: Young Link Matchup Training ==="
echo "Base model: $OPTIMIZED_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Verify optimized model exists
if [ ! -f "$OPTIMIZED_PATH" ]; then
  echo "ERROR: Optimized model not found at $OPTIMIZED_PATH"
  echo "Run yl_rl_optimization.sh first."
  exit 1
fi

# Matchup priority order
# Priority 1: Favorable (maximize wins)
# Priority 2: Common (high encounter rate)
# Priority 3: Difficult (minimize losses)

MATCHUPS="jigglypuff peach fox falco falcon marth"

for OPPONENT in $MATCHUPS; do
  TAG="${BASE_TAG}_vs_${OPPONENT}"
  OUTPUT_PATH="$OUTPUT_DIR/yl_vs_${OPPONENT}.pkl"

  echo ""
  echo "=== Training: Young Link vs $OPPONENT ==="
  echo "Output: $OUTPUT_PATH"
  echo ""

  python slippi_ai/rl/train_two.py \
    --wandb.mode=online \
    --wandb.project=project-axiom \
    --wandb.name=$TAG \
    --wandb.tags=rl,matchup,$OPPONENT,younglink,axiom \
    --config.runtime.tag=$TAG \
    \
    `# 50k steps per matchup` \
    --config.runtime.max_step=50000 \
    --config.runtime.log_interval=300 \
    \
    `# Dolphin config` \
    --config.dolphin.path="$DOLPHIN_PATH" \
    --config.dolphin.iso="$ISO_PATH" \
    --config.dolphin.console_timeout=60 \
    \
    `# Agent (Young Link)` \
    --config.agent.character=$CHAR \
    --config.agent.init_from="$OPTIMIZED_PATH" \
    \
    `# Opponent character` \
    --config.opponent.character=$OPPONENT \
    --config.opponent.type=self \
    --config.opponent.train=True \
    \
    `# Learning params` \
    --config.learner.learning_rate=2e-5 \
    --config.learner.policy_gradient_weight=6 \
    --config.learner.kl_teacher_weight=5e-4 \
    \
    `# Actor config` \
    --config.actor.rollout_length=240 \
    --config.actor.num_envs=96 \
    --config.actor.inner_batch_size=12 \
    --config.actor.async_envs=True \
    --config.actor.gpu_inference=True \
    \
    --config.save_path="$OUTPUT_PATH" \
    \
    "$@"

  echo "=== Completed: Young Link vs $OPPONENT ==="
done

echo ""
echo "=== All matchup training complete ==="
echo "Matchup models saved to: $OUTPUT_DIR"
echo ""
echo "Trained matchups:"
for OPPONENT in $MATCHUPS; do
  echo "  - yl_vs_${OPPONENT}.pkl"
done
