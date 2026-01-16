# Young Link Replay Collection Guide

Project Axiom requires high-quality Young Link competitive replays for imitation learning. This document outlines sources and collection strategies.

## The Sparse Data Challenge

Young Link is a tier-17 character with limited competitive representation:
- Fox has ~100,000+ competitive replays available
- Young Link has ~200-500 high-level replays
- Solution: Aggressive data augmentation (4-8x multiplier) + heavy RL self-play

## Primary Sources

### 1. Slippi.gg Tournament Library

The official Slippi website archives tournament replays.

**Access:**
- Visit https://slippi.gg
- Browse tournament replays
- Filter by character (Young Link)

**Quality:** High (tournament-level play)

### 2. Slippi Discord Anonymized Dataset

The Slippi Discord occasionally releases anonymized replay dumps.

**Access:**
- Join Slippi Discord: https://discord.gg/slippi
- Check #replay-sharing or #anonymized-data channels
- Filter dataset for Young Link matches

**Quality:** Variable (includes ranked matches)

### 3. Notable Young Link Players

#### Rocket (Chile)
- Highest-ranked solo Young Link (48th SSBMRank Summer 2025)
- Primary source for high-level YL play
- Search: "Rocket Young Link Melee" on YouTube/Twitch

#### Armada (Sweden, retired)
- Historical YL counterpick specialist
- Famous for YL vs Jigglypuff
- Search tournaments: Pound V, GENESIS 2, Apex 2012

#### Axe (USA)
- YL secondary
- Defeated notable players with YL
- Check major tournament VODs

#### Bambi (USA)
- 33rd at Wavelength 2024
- Active competitor

### 4. Ranked Match Archives

Slippi automatically saves ranked matches locally:
- Location: `~/Slippi/` or `Documents/Slippi/`
- Can extract YL matches from personal collection
- Lower quality but high volume

## Collection Process

### Step 1: Download Raw Replays

```bash
# Create directory structure
mkdir -p data/replays/raw

# Place .slp files in raw directory
# Files should be named descriptively:
# - tournament_player1_vs_player2_date.slp
# - ranked_yl_vs_fox_001.slp
```

### Step 2: Filter for Young Link

```bash
# Use slippi-db to filter replays
python slippi_db/parse_local.py \
    --input_dir data/replays/raw \
    --output_dir data/replays/parsed \
    --allowed_characters younglink
```

### Step 3: Apply Augmentation

```bash
# 4x data multiplier with mirror + swap
python scripts/augment_yl_data.py \
    --input data/replays/parsed \
    --output data/replays/augmented \
    --mirror \
    --player_swap
```

### Step 4: Verify Dataset

```bash
# Check file count
ls data/replays/augmented/*.parquet | wc -l

# Verify metadata
cat data/replays/augmented/meta.json
```

## Quality Guidelines

### High Priority (Tournament/Ranked)
- Major tournament sets
- Top player ranked matches
- Character specialist VODs

### Medium Priority
- Regional tournament matches
- Ranked matches against top 100 players
- Practice sets with known players

### Low Priority (Use for Volume)
- General ranked matches
- Friendlies
- Lower-level tournament pools

## Recommended Minimum Dataset

| Source | Target Count | Quality |
|--------|--------------|---------|
| Tournament replays | 100-200 | High |
| Top player ranked | 100-200 | High |
| General ranked | 200-500 | Medium |
| **Total raw** | **400-900** | - |
| **After augmentation** | **1600-3600** | - |

## Legal Considerations

- Replays are generally considered fair use for research
- Credit original players when publishing results
- Do not use replays for commercial purposes without permission
- Respect player privacy (use anonymized data when possible)

## Troubleshooting

### "No replays found"
- Verify .slp files are in the correct directory
- Check file permissions
- Ensure files aren't corrupted (try opening in Slippi Launcher)

### "Character not found"
- Young Link character ID may vary by region/version
- Check `slippi_ai/yl_rewards.py` for character ID constants
- Verify libmelee version compatibility

### "Parsing errors"
- Update peppi-py: `pip install --upgrade peppi-py`
- Check Slippi replay version compatibility
- Some very old replays may not parse correctly

## Next Steps

After collecting replays:
1. Run augmentation pipeline
2. Start imitation training: `./scripts/yl_imitation.sh`
3. Monitor training on WandB
4. Proceed to RL refinement when imitation plateaus
