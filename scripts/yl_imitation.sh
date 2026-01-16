#!/usr/bin/env sh

# Project Axiom: Young Link Imitation Learning
#
# Tuned for sparse data (~200-500 replays vs Fox's 100k+):
# - Smaller batch size to prevent overfitting
# - Reduced model capacity (2 layers, 384 hidden)
# - Lower learning rate for stability
# - More frequent evaluation checkpoints
#
# Expected runtime: 3-5 days on RTX 5090

set -e

NUM_DAYS=5
RUNTIME=$(($NUM_DAYS * 24 * 60 * 60))

# Data paths - set these environment variables before running
DATA_DIR="${DATA_ROOT:-./data/replays}/parsed"
META_PATH="${DATA_ROOT:-./data/replays}/meta.json"
MODEL_DIR="${MODEL_ROOT:-./models}/imitation"

DELAY="18"
CHAR="younglink"
TAG="axiom_yl_imitation_v1"

echo "=== Project Axiom: Young Link Imitation Learning ==="
echo "Data directory: $DATA_DIR"
echo "Output directory: $MODEL_DIR"
echo "Max runtime: $NUM_DAYS days"
echo ""

mkdir -p "$MODEL_DIR"

python scripts/train.py \
  --wandb.mode=online \
  --wandb.project=project-axiom \
  --wandb.tags=imitation,younglink,axiom \
  --config.tag=$TAG \
  --config.policy.delay=$DELAY \
  \
  `# Reduced batch size for sparse data (prevent overfitting)` \
  --config.data.batch_size=256 \
  --config.data.unroll_length=80 \
  \
  `# Conservative learning rate for sparse data` \
  --config.learner.learning_rate=5e-5 \
  --config.learner.reward_halflife=4 \
  \
  `# Reduced model capacity (2 layers instead of 3, 384 hidden instead of 512)` \
  --config.network.name=tx_like \
  --config.network.tx_like.num_layers=2 \
  --config.network.tx_like.hidden_size=384 \
  --config.network.tx_like.ffw_multiplier=2 \
  \
  `# Value function config` \
  --config.policy.train_value_head=False \
  --config.value_function.train_separate_network=True \
  --config.value_function.separate_network_config=True \
  --config.value_function.network.name=tx_like \
  --config.value_function.network.tx_like.num_layers=1 \
  --config.value_function.network.tx_like.hidden_size=384 \
  --config.value_function.network.tx_like.ffw_multiplier=2 \
  \
  `# Controller head` \
  --config.controller_head.name=autoregressive \
  --config.controller_head.autoregressive.component_depth=2 \
  --config.controller_head.autoregressive.residual_size=128 \
  \
  `# Dataset config - Young Link only` \
  --config.dataset.allowed_characters=$CHAR \
  --config.dataset.allowed_opponents=all \
  --config.dataset.data_dir=$DATA_DIR \
  --config.dataset.meta_path=$META_PATH \
  \
  `# More frequent evaluation for sparse data monitoring` \
  --config.runtime.eval_every_n=2500 \
  --config.runtime.num_eval_steps=200 \
  --config.runtime.max_runtime=$RUNTIME \
  --config.runtime.log_interval=150 \
  --config.runtime.save_interval=300 \
  \
  "$@"

echo ""
echo "=== Imitation training complete ==="
echo "Best model saved to: $MODEL_DIR"
