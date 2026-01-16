"""Young Link specific reward shaping for Project Axiom.

Extends the base reward system with YL-specific bonuses for:
- Bomb economy (pulling, hitting, z-drops)
- Projectile zoning (arrows, boomerang)
- Edgeguarding
- Recovery optimization
"""

import dataclasses
from typing import Optional

import numpy as np
import melee

from slippi_ai.types import Game, Player, Item
from slippi_ai.reward import (
    RewardConfig,
    compute_rewards,
    process_damages,
    process_deaths,
    get_bad_ledge_grabs,
    compute_approaching_factor,
    is_stalling_offstage,
    amount_offstage,
)


# Young Link character ID in Melee
YOUNG_LINK_CHARACTER_ID = melee.Character.LINK.value  # YL shares some with Link
YOUNG_LINK_ID = 22  # Young Link's actual character ID

# Action state values for Young Link special moves
# Reference: https://github.com/altf4/libmelee/blob/main/melee/enums.py
class YLActionStates:
    # Bomb (Down-B)
    DOWN_B_GROUND_START = 0x168  # Starting bomb pull
    DOWN_B_GROUND = 0x169        # Holding bomb on ground
    DOWN_B_AIR = 0x16e           # Holding bomb in air

    # Spin Attack (Up-B)
    UP_B_GROUND = 0x16f
    UP_B_AIR = 0x170

    # Bow (Neutral-B)
    NEUTRAL_B_CHARGING = 0x156
    NEUTRAL_B_ATTACKING = 0x157
    NEUTRAL_B_FULL_CHARGE = 0x158

    # Boomerang (Side-B)
    SIDE_B_GROUND = 0x15c
    SIDE_B_AIR = 0x15d

    # Z-drop related (item throw while aerial)
    ITEM_THROW_AIR = 0x2d  # Aerial item throw

    # Recovery states
    EDGE_CATCHING = melee.Action.EDGE_CATCHING.value


# Item type IDs
class ItemTypes:
    BOMB = 6  # Link/Young Link bomb
    ARROW = 7  # Fire arrow
    BOOMERANG = 8


@dataclasses.dataclass
class YLRewardConfig(RewardConfig):
    """Extended reward config with Young Link specific bonuses."""

    # Base rewards inherited from RewardConfig
    # damage_ratio: float = 0.01
    # ledge_grab_penalty: float = 0
    # approaching_factor: float = 0
    # stalling_penalty: float = 0
    # stalling_threshold: float = 20
    # nana_ratio: float = 0.5

    # Bomb economy rewards
    bomb_pull_bonus: float = 0.005       # Reward for successfully pulling a bomb
    bomb_hit_bonus: float = 0.02         # Reward for hitting opponent with bomb
    bomb_self_damage_penalty: float = 0.01  # Penalty for self-damage from bomb

    # Projectile rewards
    arrow_hit_bonus: float = 0.015       # Reward for arrow hits
    boomerang_hit_bonus: float = 0.015   # Reward for boomerang hits

    # Tech rewards
    z_drop_setup_bonus: float = 0.01     # Reward for z-drop setups

    # Edgeguard rewards
    edgeguard_attempt_bonus: float = 0.02  # Reward for going offstage when opponent is offstage
    edgeguard_success_bonus: float = 0.05  # Reward for killing during edgeguard

    # Recovery rewards
    bomb_recovery_bonus: float = 0.03    # Reward for using bomb to extend recovery
    recovery_success_bonus: float = 0.02 # Reward for successful recovery from deep offstage

    # Anti-patterns
    predictable_recovery_penalty: float = 0.015  # Penalty for always recovering the same way


def is_young_link(player: Player) -> np.ndarray:
    """Check if player is Young Link."""
    return player.character == YOUNG_LINK_ID


def detected_bomb_pull(player: Player) -> np.ndarray:
    """Detect when Young Link successfully completes a bomb pull.

    Bomb pull takes 40 frames total, bomb is grabbable at frame 16.
    We detect the transition into holding state.
    """
    # Check for transition into bomb holding state
    was_pulling = np.isin(
        player.action[:-1],
        [YLActionStates.DOWN_B_GROUND_START]
    )
    now_holding = np.isin(
        player.action[1:],
        [YLActionStates.DOWN_B_GROUND, YLActionStates.DOWN_B_AIR]
    )
    return np.logical_and(was_pulling, now_holding)


def detected_bomb_throw(player: Player) -> np.ndarray:
    """Detect when Young Link throws a bomb."""
    was_holding = np.isin(
        player.action[:-1],
        [YLActionStates.DOWN_B_GROUND, YLActionStates.DOWN_B_AIR]
    )
    # Transition out of holding state (could be throw, z-drop, etc.)
    now_not_holding = ~np.isin(
        player.action[1:],
        [YLActionStates.DOWN_B_GROUND, YLActionStates.DOWN_B_AIR]
    )
    return np.logical_and(was_holding, now_not_holding)


def detected_z_drop(player: Player) -> np.ndarray:
    """Detect z-drop (aerial bomb drop while in air).

    Z-drop is dropping bomb in air without throwing - allows for quick followups.
    """
    was_holding_air = player.action[:-1] == YLActionStates.DOWN_B_AIR
    not_on_ground = ~player.on_ground[:-1]

    # Z-drop transitions to a different state than normal throw
    is_z_drop = np.logical_and(was_holding_air, not_on_ground)
    is_z_drop = np.logical_and(is_z_drop, detected_bomb_throw(player))

    return is_z_drop


def detected_arrow_shot(player: Player) -> np.ndarray:
    """Detect when Young Link fires an arrow."""
    was_charging = np.isin(
        player.action[:-1],
        [YLActionStates.NEUTRAL_B_CHARGING, YLActionStates.NEUTRAL_B_FULL_CHARGE]
    )
    now_attacking = player.action[1:] == YLActionStates.NEUTRAL_B_ATTACKING
    return np.logical_and(was_charging, now_attacking)


def detected_boomerang_throw(player: Player) -> np.ndarray:
    """Detect when Young Link throws boomerang."""
    # Transition into boomerang throw state
    was_not_boomerang = ~np.isin(
        player.action[:-1],
        [YLActionStates.SIDE_B_GROUND, YLActionStates.SIDE_B_AIR]
    )
    now_boomerang = np.isin(
        player.action[1:],
        [YLActionStates.SIDE_B_GROUND, YLActionStates.SIDE_B_AIR]
    )
    return np.logical_and(was_not_boomerang, now_boomerang)


def is_offstage(player: Player, stage: np.ndarray, threshold: float = 10) -> np.ndarray:
    """Check if player is offstage (past ledge)."""
    return amount_offstage(player, stage) > threshold


def in_edgeguard_position(
    player: Player,
    opponent: Player,
    stage: np.ndarray,
) -> np.ndarray:
    """Detect when YL is in edgeguard position (opponent offstage, YL near edge or offstage)."""
    opponent_offstage = is_offstage(opponent, stage, threshold=5)
    player_near_edge_or_offstage = is_offstage(player, stage, threshold=0)

    return np.logical_and(opponent_offstage[1:], player_near_edge_or_offstage[1:])


def detected_edgeguard_kill(
    player: Player,
    opponent: Player,
    stage: np.ndarray,
) -> np.ndarray:
    """Detect when opponent dies while offstage (edgeguard success)."""
    opponent_offstage = is_offstage(opponent, stage, threshold=5)
    opponent_deaths = process_deaths(opponent.action)

    return np.logical_and(opponent_offstage[:-1], opponent_deaths)


def detected_recovery_from_deep(
    player: Player,
    stage: np.ndarray,
    deep_threshold: float = 40,
) -> np.ndarray:
    """Detect successful recovery from deep offstage."""
    was_deep_offstage = amount_offstage(player, stage)[:-1] > deep_threshold
    now_onstage = amount_offstage(player, stage)[1:] < 5

    return np.logical_and(was_deep_offstage, now_onstage)


def compute_yl_rewards(
    game: Game,
    config: YLRewardConfig,
) -> np.ndarray:
    """Compute rewards with Young Link specific bonuses.

    Args:
        game: Game state data (nest of np.arrays of length T)
        config: YL reward configuration

    Returns:
        A length (T-1) np.array of rewards
    """
    # Start with base rewards
    base_rewards = compute_rewards(
        game,
        damage_ratio=config.damage_ratio,
        ledge_grab_penalty=config.ledge_grab_penalty,
        approaching_factor=config.approaching_factor,
        stalling_penalty=config.stalling_penalty,
        stalling_threshold=config.stalling_threshold,
        nana_ratio=config.nana_ratio,
    )

    rewards = base_rewards.copy()

    # Only apply YL bonuses if player is Young Link
    p0_is_yl = is_young_link(game.p0)
    p1_is_yl = is_young_link(game.p1)

    def apply_yl_bonuses(player: Player, opponent: Player, sign: float):
        """Apply YL-specific bonuses for one player."""
        nonlocal rewards

        # Bomb pull bonus
        bomb_pulls = detected_bomb_pull(player).astype(np.float32)
        rewards += sign * config.bomb_pull_bonus * bomb_pulls

        # Z-drop setup bonus
        z_drops = detected_z_drop(player).astype(np.float32)
        rewards += sign * config.z_drop_setup_bonus * z_drops

        # Arrow shot bonus (hit detection would need damage tracking)
        arrow_shots = detected_arrow_shot(player).astype(np.float32)
        rewards += sign * config.arrow_hit_bonus * arrow_shots * 0.5  # Partial credit for shooting

        # Boomerang throw bonus
        boomerang_throws = detected_boomerang_throw(player).astype(np.float32)
        rewards += sign * config.boomerang_hit_bonus * boomerang_throws * 0.5

        # Edgeguard attempt bonus
        edgeguard_attempts = in_edgeguard_position(player, opponent, game.stage).astype(np.float32)
        rewards += sign * config.edgeguard_attempt_bonus * edgeguard_attempts

        # Edgeguard success bonus
        edgeguard_kills = detected_edgeguard_kill(player, opponent, game.stage).astype(np.float32)
        rewards += sign * config.edgeguard_success_bonus * edgeguard_kills

        # Recovery success bonus
        recovery_success = detected_recovery_from_deep(player, game.stage).astype(np.float32)
        rewards += sign * config.recovery_success_bonus * recovery_success

    # Apply bonuses for p0 if they're YL (positive sign)
    if np.any(p0_is_yl):
        apply_yl_bonuses(game.p0, game.p1, sign=1.0)

    # Apply bonuses for p1 if they're YL (negative sign for zero-sum)
    if np.any(p1_is_yl):
        apply_yl_bonuses(game.p1, game.p0, sign=-1.0)

    return rewards


def yl_player_stats(
    player: Player,
    opponent: Player,
    stage: np.ndarray,
) -> dict:
    """Compute Young Link specific statistics."""
    FPM = 60 * 60  # Frames per minute

    stats = {}

    if np.any(is_young_link(player)):
        stats.update({
            'bomb_pulls': detected_bomb_pull(player).sum(),
            'z_drops': detected_z_drop(player).sum(),
            'arrow_shots': detected_arrow_shot(player).sum(),
            'boomerang_throws': detected_boomerang_throw(player).sum(),
            'edgeguard_attempts': in_edgeguard_position(player, opponent, stage).sum(),
            'edgeguard_kills': detected_edgeguard_kill(player, opponent, stage).sum(),
            'deep_recoveries': detected_recovery_from_deep(player, stage).sum(),
            'bomb_pulls_per_minute': detected_bomb_pull(player).mean() * FPM,
        })

    return stats


# Default YL reward config optimized for competitive play
DEFAULT_YL_CONFIG = YLRewardConfig(
    # Base rewards
    damage_ratio=0.01,
    ledge_grab_penalty=0.02,
    approaching_factor=0.001,
    stalling_penalty=0.01,
    stalling_threshold=20,
    nana_ratio=0.5,

    # YL-specific (tuned for competitive emphasis)
    bomb_pull_bonus=0.005,
    bomb_hit_bonus=0.02,
    bomb_self_damage_penalty=0.01,
    arrow_hit_bonus=0.015,
    boomerang_hit_bonus=0.015,
    z_drop_setup_bonus=0.01,
    edgeguard_attempt_bonus=0.02,
    edgeguard_success_bonus=0.05,
    bomb_recovery_bonus=0.03,
    recovery_success_bonus=0.02,
    predictable_recovery_penalty=0.015,
)


# Aggressive config for discovering novel bomb-heavy strategies
AGGRESSIVE_BOMB_CONFIG = YLRewardConfig(
    damage_ratio=0.01,
    ledge_grab_penalty=0.02,
    approaching_factor=0.002,  # More aggressive approaching
    stalling_penalty=0.02,     # Higher stalling penalty
    stalling_threshold=15,
    nana_ratio=0.5,

    # Heavy bomb emphasis
    bomb_pull_bonus=0.01,      # Double bomb pull reward
    bomb_hit_bonus=0.04,       # Double bomb hit reward
    bomb_self_damage_penalty=0.005,  # Lower self-damage penalty (worth the trade)
    arrow_hit_bonus=0.01,
    boomerang_hit_bonus=0.01,
    z_drop_setup_bonus=0.02,   # Double z-drop reward
    edgeguard_attempt_bonus=0.03,
    edgeguard_success_bonus=0.08,
    bomb_recovery_bonus=0.05,
    recovery_success_bonus=0.03,
    predictable_recovery_penalty=0.02,
)


# Defensive/zoning config for matchups vs rushdown characters
ZONING_CONFIG = YLRewardConfig(
    damage_ratio=0.01,
    ledge_grab_penalty=0.01,
    approaching_factor=0.0,    # Don't reward approaching
    stalling_penalty=0.005,    # Lower stalling penalty (camping is ok)
    stalling_threshold=25,
    nana_ratio=0.5,

    # Projectile emphasis
    bomb_pull_bonus=0.008,
    bomb_hit_bonus=0.025,
    bomb_self_damage_penalty=0.015,
    arrow_hit_bonus=0.025,     # High arrow reward
    boomerang_hit_bonus=0.025, # High boomerang reward
    z_drop_setup_bonus=0.015,
    edgeguard_attempt_bonus=0.015,
    edgeguard_success_bonus=0.04,
    bomb_recovery_bonus=0.03,
    recovery_success_bonus=0.025,
    predictable_recovery_penalty=0.01,
)
