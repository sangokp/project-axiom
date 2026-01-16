#!/usr/bin/env bash

# Project Axiom: Replay Download Script
#
# Downloads tournament replay archives from known public sources.
# Run this first, then filter for Young Link matches.
#
# Sources (from Perplexity research):
# - Slippi Google Cloud Storage tournament archives
# - Community-shared bulk datasets

set -e

RAW_DIR="${1:-./data/replays/raw}"
TEMP_DIR="./data/replays/.downloads"

echo "=== Project Axiom: Replay Download ==="
echo "Output directory: $RAW_DIR"
echo ""

mkdir -p "$RAW_DIR/tournaments"
mkdir -p "$TEMP_DIR"

# Known tournament archive URLs from Slippi GCS
# Source: https://www.reddit.com/r/SSBM/comments/dca3ag/
TOURNAMENT_URLS=(
    "https://storage.googleapis.com/slippi.appspot.com/dump/gang-replays.7z"
    "https://storage.googleapis.com/slippi.appspot.com/dump/full-bloom-5-replays.7z"
    "https://storage.googleapis.com/slippi.appspot.com/dump/pound-2019-replays.7z"
)

echo "=== Downloading Tournament Archives ==="
echo ""

for URL in "${TOURNAMENT_URLS[@]}"; do
    FILENAME=$(basename "$URL")
    DEST="$TEMP_DIR/$FILENAME"

    if [ -f "$DEST" ]; then
        echo "[SKIP] $FILENAME already downloaded"
    else
        echo "[DOWNLOAD] $FILENAME"
        curl -L -o "$DEST" "$URL" || echo "[WARN] Failed to download $URL"
    fi
done

echo ""
echo "=== Extracting Archives ==="
echo ""

# Check if 7z is available
if ! command -v 7z &> /dev/null; then
    echo "7z not found. Installing via Homebrew..."
    brew install p7zip || echo "[WARN] Could not install p7zip"
fi

for ARCHIVE in "$TEMP_DIR"/*.7z; do
    if [ -f "$ARCHIVE" ]; then
        BASENAME=$(basename "$ARCHIVE" .7z)
        EXTRACT_DIR="$RAW_DIR/tournaments/$BASENAME"

        if [ -d "$EXTRACT_DIR" ]; then
            echo "[SKIP] $BASENAME already extracted"
        else
            echo "[EXTRACT] $BASENAME"
            mkdir -p "$EXTRACT_DIR"
            7z x -o"$EXTRACT_DIR" "$ARCHIVE" || echo "[WARN] Failed to extract $ARCHIVE"
        fi
    fi
done

echo ""
echo "=== Download Summary ==="
echo ""

# Count .slp files
SLP_COUNT=$(find "$RAW_DIR" -name "*.slp" 2>/dev/null | wc -l | tr -d ' ')
echo "Total .slp files downloaded: $SLP_COUNT"

echo ""
echo "=== Next Steps ==="
echo ""
echo "1. Join Slippi Discord (https://discord.gg/slippi) for the 95k anonymized dataset"
echo "2. Filter for Young Link:"
echo "   cd ~/Projects/project-axiom"
echo "   source .venv/bin/activate"
echo "   python slippi_db/parse_local.py \\"
echo "       --input_dir data/replays/raw \\"
echo "       --output_dir data/replays/parsed \\"
echo "       --allowed_characters younglink"
echo ""
echo "3. After filtering, run augmentation:"
echo "   python scripts/augment_yl_data.py \\"
echo "       --input data/replays/parsed \\"
echo "       --output data/replays/augmented \\"
echo "       --mirror --player_swap"
