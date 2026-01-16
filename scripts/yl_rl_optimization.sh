#!/usr/bin/env sh

# Project Axiom: Young Link RL Optimization (Phase 2)
#
# Goal: Push beyond human play by reducing teacher constraint
# - Lower KL teacher weight allows novel strategy discovery
# - Higher policy gradient weight for more exploration
# - Enhanced YL-specific rewards for bomb/projectile play
#
# This is where we find Young Link's TRUE CEILING.
#
# Expected runtime: 7-14 days on RTX 5090
# Prerequisites: Completed RL foundation (yl_rl_foundation.sh)

set -e

CHAR=younglink
DELAY=18
TAG="axiom_yl_rl_optimization_v1"

# Paths
DOLPHIN_PATH="${DOLPHIN_PATH:-/Applications/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin}"
ISO_PATH="${ISO_PATH:-./SSBM.iso}"
FOUNDATION_PATH="${MODEL_ROOT:-./models}/rl_foundation/axiom_yl_rl_foundation_best.pkl"
TEACHER_PATH="${MODEL_ROOT:-./models}/imitation/axiom_yl_imitation_best.pkl"
OUTPUT_DIR="${MODEL_ROOT:-./models}/rl_optimized"

echo "=== Project Axiom: Young Link RL Optimization (Phase 2) ==="
echo "Foundation model: $FOUNDATION_PATH"
echo "Teacher model: $TEACHER_PATH"
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "NOTE: This phase reduces teacher constraint to allow novel discovery."
echo "      The bot may develop strategies humans haven't explored."
echo ""

mkdir -p "$OUTPUT_DIR"

# Verify foundation model exists
if [ ! -f "$FOUNDATION_PATH" ]; then
  echo "ERROR: Foundation model not found at $FOUNDATION_PATH"
  echo "Run yl_rl_foundation.sh first."
  exit 1
fi

python slippi_ai/rl/run.py \
  --wandb.mode=online \
  --wandb.project=project-axiom \
  --wandb.name=$TAG \
  --wandb.tags=rl,optimization,younglink,axiom,ceiling-push \
  --config.runtime.tag=$TAG \
  \
  `# Extended training - 200k steps for optimization` \
  --config.runtime.max_step=200000 \
  --config.runtime.log_interval=300 \
  \
  `# Dolphin config` \
  --config.dolphin.path="$DOLPHIN_PATH" \
  --config.dolphin.iso="$ISO_PATH" \
  --config.dolphin.console_timeout=60 \
  \
  `# Learning parameters - slightly higher LR for continued learning` \
  --config.learner.learning_rate=2e-5 \
  --config.learner.value_cost=1 \
  --config.learner.reward_halflife=4 \
  \
  `# Enhanced YL-specific rewards for optimization phase` \
  --config.learner.reward.damage_ratio=0.01 \
  --config.learner.reward.ledge_grab_penalty=0.02 \
  \
  `# KEY CHANGE: Reduced teacher constraint for novel discovery` \
  --config.learner.policy_gradient_weight=8 \
  --config.learner.kl_teacher_weight=1e-4 \
  \
  `# PPO with slightly more exploration` \
  --config.learner.ppo.num_epochs=2 \
  --config.learner.ppo.num_batches=16 \
  --config.learner.ppo.beta=4e-1 \
  --config.learner.ppo.epsilon=1e-2 \
  --config.learner.ppo.minibatched=False \
  \
  `# Teacher model (original imitation, not foundation)` \
  --config.teacher="$TEACHER_PATH" \
  \
  `# Initialize from foundation model` \
  --config.init_from="$FOUNDATION_PATH" \
  \
  `# Self-play opponent` \
  --config.opponent.type=self \
  --config.opponent.train=True \
  \
  `# Actor config - RTX 5090 optimized` \
  --config.actor.rollout_length=240 \
  --config.actor.num_envs=128 \
  --config.actor.inner_batch_size=16 \
  --config.actor.async_envs=True \
  --config.actor.num_env_steps=4 \
  --config.actor.gpu_inference=True \
  \
  `# Agent config` \
  --config.agent.character=$CHAR \
  --config.agent.batch_steps=4 \
  \
  `# Reset config` \
  --config.runtime.reset_every_n_steps=512 \
  --config.runtime.burnin_steps_after_reset=5 \
  --config.optimizer_burnin_steps=128 \
  --config.value_burnin_steps=128 \
  \
  "$@"

echo ""
echo "=== RL Optimization training complete ==="
echo "Optimized model saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Evaluate with: python scripts/eval_two.py --p2.ai.path=$OUTPUT_DIR/best.pkl"
echo "  2. Train matchup-specific models with: ./scripts/yl_matchup_training.sh"
