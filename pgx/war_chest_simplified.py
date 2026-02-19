from typing import Optional

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._src.games.war_chest_simplified import (
    Game,
    GameState,
    NUM_ACTIONS,
    NUM_HEXES,
    NUM_UNIT_TYPES,
    NUM_COLORED_UNITS,
    NUM_HEX_CHANNELS,
    NUM_GLOBAL_FEATURES,
    OBS_SIZE,
    MAX_STEP_COUNT,
    ACT_DEPLOY_START,
    ACT_DEPLOY_END,
    ACT_MOVE_START,
    ACT_MOVE_END,
    ACT_ATTACK_START,
    ACT_ATTACK_END,
    ACT_CONTROL_START,
    ACT_CONTROL_END,
    ACT_RECRUIT_START,
    ACT_RECRUIT_END,
    ACT_INITIATIVE_START,
    ACT_INITIATIVE_END,
    ACT_PASS_START,
    ACT_PASS_END,
    NEIGHBOR_MAP,
)
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    """War Chest Simplified game state for PGX.
    
    This wraps the internal GameState with the standard PGX State interface.
    
    The internal GameState (a NamedTuple with board, bags, etc.) is stored 
    in the _x field, following the same pattern as other PGX games.
    """
    current_player: Array = jnp.int32(0)
    observation: Array = jnp.zeros(OBS_SIZE, dtype=jnp.float32)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.ones(NUM_ACTIONS, dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    # --- War Chest Simplified specific ---
    _x: GameState = GameState() # Renamed from GameStateNT

    @property
    def env_id(self) -> core.EnvId:
        return "war_chest_simplified"


class WarChestSimplified(core.Env):
    """War Chest Simplified environment.
    
    A simplified version of the War Chest board game, implemented as a JAX-native
    environment for parallel execution on accelerators.
    
    Key mechanics:
    - Hex-grid board with 37 hexes (radius 3)
    - 4 unit types (colors) per player + 1 royal coin (can only be discarded face-down)
    - Bag-building: coins cycle between bag, hand, discard (face-up/down), and board
    - Cemetery: coins removed by attacks are permanently out of the game
    - Multiple action types: Deploy/Bolster, Move, Attack, Control, Recruit, Initiative, Pass
    - Victory: Place all 6 control markers
    
    Example usage:
        >>> import pgx
        >>> env = pgx.make("war_chest_simplified")
        >>> state = env.init(jax.random.PRNGKey(0))
        >>> action = jnp.int32(0)  # Some valid action
        >>> state = env.step(state, action, jax.random.PRNGKey(1))
    """
    
    def __init__(self):
        super().__init__()
        self._game = Game()
    
    def step(
        self,
        state: core.State,
        action: Array,
        key: Optional[Array] = None,
    ) -> core.State:
        """Take a step in the environment.
        
        This game requires randomness for drawing coins from bags,
        so a PRNGKey must be provided.
        
        Args:
            state: Current game state
            action: Action to take (0-658)
            key: PRNGKey for randomness (required)
            
        Returns:
            New game state after the action
        """
        assert key is not None, (
            "War Chest Simplified requires a PRNGKey for randomness. "
            "Please specify PRNGKey at the third argument:\n\n"
            "  state = env.step(state, action, key)\n"
        )
        return super().step(state, action, key)
    
    def _init(self, key: PRNGKey) -> State:
        """Initialize a new game.
        
        Args:
            key: PRNGKey for random initialization (shuffle bags, determine first player)
            
        Returns:
            Initial game state
        """
        x = self._game.init(key)
        current_player = x.current_player
        legal_action_mask = self._game.legal_action_mask(x)
        
        return State(  # type: ignore
            current_player=current_player,
            legal_action_mask=legal_action_mask,
            _x=x,
        )
    
    def _step(self, state: core.State, action: Array, key) -> State:
        """Execute one step of the game.
        
        Args:
            state: Current state
            action: Action to execute
            key: PRNGKey for randomness (coin shuffling)
            
        Returns:
            New state after action
        """
        assert isinstance(state, State)
        
        x = self._game.step(state._x, action, key)
        
        terminated = self._game.is_terminal(x)
        rewards = self._game.rewards(x)
        legal_action_mask = self._game.legal_action_mask(x)
        
        # Handle max steps truncation (using PGX _step_count)
        truncated = (state._step_count >= MAX_STEP_COUNT) & ~x.terminated
        
        return state.replace(  # type: ignore
            current_player=x.current_player,
            legal_action_mask=legal_action_mask,
            terminated=terminated,
            truncated=truncated,
            rewards=rewards,
            _x=x,
        )
    
    def _observe(self, state: core.State, player_id: Array) -> Array:
        """Generate observation for a specific player.
        
        The observation includes:
        - Board state (units, stacks, control markers) from player's perspective
        - Hand contents (private information, histogram)
        - Opponent's face-up discard (public information)
        - Supply counts (public)
        - Cemetery counts (public)
        - Turn and initiative indicators
        
        Args:
            state: Current game state
            player_id: Player to generate observation for (0 or 1)
            
        Returns:
            Observation array of shape (OBS_SIZE,) = (459,)
        """
        assert isinstance(state, State)
        return self._game.observe(state._x, player_id)
    
    @property
    def id(self) -> core.EnvId:
        return "war_chest_simplified"
    
    @property
    def version(self) -> str:
        return "v1"  # Updated version for merged implementation
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def _illegal_action_penalty(self) -> float:
        """Penalty for taking an illegal action."""
        return -1.0


# =============================================================================
# Helper Functions for Game Analysis
# =============================================================================

def action_to_string(action: int) -> str:
    """Convert action index to human-readable string.
    
    Args:
        action: Action index (0-658)
        
    Returns:
        Human-readable description of the action
    """
    if action < ACT_DEPLOY_END:
        rel_idx = action - ACT_DEPLOY_START
        hex_idx = rel_idx // NUM_COLORED_UNITS
        deploy_unit = rel_idx % NUM_COLORED_UNITS
        units = ["Red", "Green", "Blue", "Yellow"]
        return f"Deploy/Bolster {units[deploy_unit]} at hex {hex_idx}"
    
    elif action < ACT_MOVE_END:
        rel_idx = action - ACT_MOVE_START
        src_hex = rel_idx // 6
        direction = rel_idx % 6
        dirs = ["E", "NE", "NW", "W", "SW", "SE"]
        return f"Move from hex {src_hex} direction {dirs[direction]}"
    
    elif action < ACT_ATTACK_END:
        rel_idx = action - ACT_ATTACK_START
        src_hex = rel_idx // 6
        direction = rel_idx % 6
        dirs = ["E", "NE", "NW", "W", "SW", "SE"]
        return f"Attack from hex {src_hex} direction {dirs[direction]}"
    
    elif action < ACT_CONTROL_END:
        hex_idx = action - ACT_CONTROL_START
        return f"Control hex {hex_idx}"
    
    elif action < ACT_RECRUIT_END:
        rel_idx = action - ACT_RECRUIT_START
        recruit_target = rel_idx // NUM_UNIT_TYPES
        discard_coin = rel_idx % NUM_UNIT_TYPES
        units = ["Red", "Green", "Blue", "Yellow", "Royal"]
        return f"Recruit {units[recruit_target]} (discard {units[discard_coin]})"
    
    elif action < ACT_INITIATIVE_END:
        discard_coin = action - ACT_INITIATIVE_START
        units = ["Red", "Green", "Blue", "Yellow", "Royal"]
        return f"Claim Initiative (discard {units[discard_coin]})"
    
    elif action < ACT_PASS_END:
        discard_coin = action - ACT_PASS_START
        units = ["Red", "Green", "Blue", "Yellow", "Royal"]
        return f"Pass (discard {units[discard_coin]})"
    
    return f"Unknown action {action}"


def get_hex_neighbor(hex_idx: int, direction: int) -> int:
    """Get the neighbor of a hex in a given direction.
    
    Args:
        hex_idx: Source hex index (0-36)
        direction: Direction (0-5): E, NE, NW, W, SW, SE
        
    Returns:
        Neighbor hex index, or -1 if off-board
    """
    return int(NEIGHBOR_MAP[hex_idx, direction])


def state_summary(state: State) -> str:
    """Generate a human-readable summary of the game state.
    
    Args:
        state: Current game state
        
    Returns:
        Multi-line string describing the state
    """
    x = state._x
    lines = []
    
    lines.append(f"Step: {state._step_count}, Phase: {'Draw' if x.phase == 0 else 'Action'}")
    lines.append(f"Current Player: P{x.current_player}, Initiative: P{x.initiative_player}")
    lines.append(f"Turn in round: {x.turn_in_round}")
    
    # Control markers
    p0_control = jnp.sum(x.control_owner == 0)
    p1_control = jnp.sum(x.control_owner == 1)
    lines.append(f"Control: P0={p0_control}/6, P1={p1_control}/6")
    
    # Player resources
    for p in range(2):
        lines.append(f"\nPlayer {p}:")
        lines.append(f"  Hand: {x.hand[p].tolist()}")
        lines.append(f"  Bag: {x.bag[p].tolist()}")
        lines.append(f"  Supply: {x.supply[p].tolist()}")
        lines.append(f"  Discard (up): {x.discard_faceup[p].tolist()}")
        lines.append(f"  Discard (down): {x.discard_facedown[p].tolist()}")
        lines.append(f"  Cemetery: {x.cemetery[p].tolist()}")
    
    # Units on board
    lines.append("\nUnits on board:")
    for h in range(NUM_HEXES):
        if x.board_unit_type[h] >= 0:
            units = ["Red", "Green", "Blue", "Yellow", "Royal"]
            lines.append(f"  Hex {h}: P{x.board_owner[h]} {units[x.board_unit_type[h]]} x{x.board_count[h]}")
    
    if x.terminated:
        lines.append(f"\nGame Over! Winner: P{x.winner}")
    
    return "\n".join(lines)
