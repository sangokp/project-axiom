#!/usr/bin/env python3
"""
Project Axiom: Young Link Replay Filter

Filters raw .slp replay files to extract only Young Link matches.
Uses py-slippi for stable parsing (peppi-py has Rust panic issues).

Usage:
    python scripts/filter_young_link.py \
        --input data/replays/raw \
        --output data/replays/parsed \
        --min_frames 1800  # ~30 seconds minimum game length
"""

import argparse
import json
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import sys
import traceback

try:
    from slippi import Game
    from slippi.id import CSSCharacter
except ImportError:
    print("py-slippi not installed. Run: pip install py-slippi")
    sys.exit(1)


# Young Link character ID in py-slippi
YOUNG_LINK_CSS_ID = 21  # CSSCharacter enum value for Young Link


@dataclass
class ReplayMetadata:
    """Metadata for a filtered replay."""
    filename: str
    stage: int
    duration_frames: int
    yl_port: int
    yl_player_code: Optional[str]
    opponent_character: int
    opponent_player_code: Optional[str]
    date: Optional[str]


def is_young_link(char_id) -> bool:
    """Check if character is Young Link."""
    if char_id is None:
        return False
    # Handle both enum and int
    if hasattr(char_id, 'value'):
        return char_id.value == YOUNG_LINK_CSS_ID
    return char_id == YOUNG_LINK_CSS_ID


def parse_replay(filepath: Path) -> Optional[dict]:
    """Parse a replay file and extract relevant info."""
    try:
        # Skip tiny files
        if filepath.stat().st_size < 1000:
            return None

        game = Game(str(filepath))

        if game.start is None:
            return None

        # Get players
        players = game.start.players
        if players is None:
            return None

        active_players = [p for p in players if p is not None]
        if len(active_players) < 2:
            return None

        # Get frame count
        frame_count = 0
        if game.frames is not None:
            frame_count = len(game.frames)
        elif game.metadata is not None and game.metadata.duration is not None:
            frame_count = game.metadata.duration

        # Get stage
        stage = 0
        if game.start.stage is not None:
            stage = game.start.stage.value if hasattr(game.start.stage, 'value') else int(game.start.stage)

        # Get date
        date = None
        if game.metadata is not None and game.metadata.date is not None:
            date = str(game.metadata.date)

        return {
            'players': active_players,
            'frame_count': frame_count,
            'stage': stage,
            'date': date,
        }

    except Exception:
        return None


def filter_replays(
    input_dir: Path,
    output_dir: Path,
    min_frames: int = 1800,
    verbose: bool = True,
) -> list[ReplayMetadata]:
    """Filter replays for Young Link matches."""

    output_dir.mkdir(parents=True, exist_ok=True)

    slp_files = list(input_dir.rglob("*.slp"))
    print(f"Found {len(slp_files)} .slp files to process")
    print()

    filtered = []
    skipped_no_yl = 0
    skipped_too_short = 0
    skipped_parse_error = 0

    for i, filepath in enumerate(slp_files):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(slp_files)}... (found {len(filtered)} YL matches)")

        # Parse replay
        data = parse_replay(filepath)
        if data is None:
            skipped_parse_error += 1
            continue

        # Check frame count
        if data['frame_count'] < min_frames:
            skipped_too_short += 1
            continue

        # Check for Young Link
        players = data['players']
        yl_port = None
        yl_player_code = None
        opponent_char = None
        opponent_code = None

        for idx, player in enumerate(players):
            if player is None:
                continue
            char = player.character
            if is_young_link(char):
                yl_port = idx
                # py-slippi uses tag instead of netplay
                yl_player_code = getattr(player, 'tag', None)
            else:
                if char is not None:
                    opponent_char = char.value if hasattr(char, 'value') else int(char)
                opponent_code = getattr(player, 'tag', None)

        if yl_port is None:
            skipped_no_yl += 1
            continue

        # Copy file to output
        dest = output_dir / filepath.name
        if dest.exists():
            # Add suffix if duplicate name
            dest = output_dir / f"{filepath.stem}_{i}{filepath.suffix}"

        shutil.copy(filepath, dest)

        # Record metadata
        meta = ReplayMetadata(
            filename=dest.name,
            stage=data['stage'],
            duration_frames=data['frame_count'],
            yl_port=yl_port,
            yl_player_code=yl_player_code,
            opponent_character=opponent_char or 0,
            opponent_player_code=opponent_code,
            date=data['date'],
        )
        filtered.append(meta)

    print()
    print("=== Filter Results ===")
    print(f"Total processed: {len(slp_files)}")
    print(f"Young Link matches found: {len(filtered)}")
    print(f"Skipped (no Young Link): {skipped_no_yl}")
    print(f"Skipped (too short): {skipped_too_short}")
    print(f"Skipped (parse error): {skipped_parse_error}")

    return filtered


def save_metadata(filtered: list[ReplayMetadata], output_dir: Path):
    """Save metadata JSON for filtered replays."""
    meta_path = output_dir / "meta.json"

    meta = {
        'character': 'younglink',
        'count': len(filtered),
        'replays': [asdict(r) for r in filtered],
    }

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Metadata saved to: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter replays for Young Link matches"
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input directory containing raw .slp files'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for filtered replays'
    )
    parser.add_argument(
        '--min_frames',
        type=int,
        default=1800,
        help='Minimum game length in frames (default: 1800 = ~30 seconds)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    print("=== Project Axiom: Young Link Filter ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Min frames: {args.min_frames}")
    print()

    filtered = filter_replays(
        args.input,
        args.output,
        min_frames=args.min_frames,
        verbose=not args.quiet,
    )

    if filtered:
        save_metadata(filtered, args.output)
        print()
        print("=== Next Steps ===")
        print(f"  python scripts/augment_yl_data.py \\")
        print(f"      --input {args.output} \\")
        print(f"      --output data/replays/augmented \\")
        print(f"      --mirror --player_swap")
    else:
        print()
        print("[WARN] No Young Link replays found!")
        print("Check that your input directory contains .slp files with Young Link matches.")


if __name__ == "__main__":
    main()
