#!/usr/bin/env sh

# Project Axiom: Young Link RL Foundation Training (Phase 1)
#
# Goal: Build solid foundation by staying close to imitation teacher
# - Higher KL teacher weight to prevent divergence
# - Standard PPO parameters
# - YL-specific reward shaping enabled
#
# Expected runtime: 5-7 days on RTX 5090
# Prerequisites: Completed imitation learning (yl_imitation.sh)

set -e

CHAR=younglink
DELAY=18
TAG="axiom_yl_rl_foundation_v1"

# Paths - set these environment variables
DOLPHIN_PATH="${DOLPHIN_PATH:-/Applications/Slippi Dolphin.app/Contents/MacOS/Slippi Dolphin}"
ISO_PATH="${ISO_PATH:-./SSBM.iso}"
TEACHER_PATH="${MODEL_ROOT:-./models}/imitation/axiom_yl_imitation_best.pkl"
OUTPUT_DIR="${MODEL_ROOT:-./models}/rl_foundation"

echo "=== Project Axiom: Young Link RL Foundation (Phase 1) ==="
echo "Teacher model: $TEACHER_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Dolphin path: $DOLPHIN_PATH"
echo ""

mkdir -p "$OUTPUT_DIR"

# Verify teacher model exists
if [ ! -f "$TEACHER_PATH" ]; then
  echo "ERROR: Teacher model not found at $TEACHER_PATH"
  echo "Run yl_imitation.sh first to train the imitation model."
  exit 1
fi

python slippi_ai/rl/run.py \
  --wandb.mode=online \
  --wandb.project=project-axiom \
  --wandb.name=$TAG \
  --wandb.tags=rl,foundation,younglink,axiom \
  --config.runtime.tag=$TAG \
  \
  `# Training duration - 100k steps` \
  --config.runtime.max_step=100000 \
  --config.runtime.log_interval=300 \
  \
  `# Dolphin config` \
  --config.dolphin.path="$DOLPHIN_PATH" \
  --config.dolphin.iso="$ISO_PATH" \
  --config.dolphin.console_timeout=60 \
  \
  `# Learning parameters` \
  --config.learner.learning_rate=3e-5 \
  --config.learner.value_cost=1 \
  --config.learner.reward_halflife=4 \
  \
  `# YL-specific reward shaping` \
  --config.learner.reward.damage_ratio=0.01 \
  --config.learner.reward.ledge_grab_penalty=0.02 \
  \
  `# Policy gradient weight and teacher constraint` \
  --config.learner.policy_gradient_weight=5 \
  --config.learner.kl_teacher_weight=3e-3 \
  \
  `# PPO hyperparameters` \
  --config.learner.ppo.num_epochs=2 \
  --config.learner.ppo.num_batches=16 \
  --config.learner.ppo.beta=3e-1 \
  --config.learner.ppo.epsilon=1e-2 \
  --config.learner.ppo.minibatched=False \
  \
  `# Teacher model (from imitation)` \
  --config.teacher="$TEACHER_PATH" \
  \
  `# Self-play opponent` \
  --config.opponent.type=self \
  --config.opponent.train=True \
  \
  `# Actor config - RTX 5090 optimized (128 envs, 16 batch)` \
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
  `# Reset config for stability` \
  --config.runtime.reset_every_n_steps=512 \
  --config.runtime.burnin_steps_after_reset=5 \
  --config.optimizer_burnin_steps=128 \
  --config.value_burnin_steps=128 \
  \
  "$@"

echo ""
echo "=== RL Foundation training complete ==="
echo "Model saved to: $OUTPUT_DIR"
