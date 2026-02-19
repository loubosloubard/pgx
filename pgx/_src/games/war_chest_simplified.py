from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax import Array

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

# =============================================================================
# GAME CONSTANTS
# =============================================================================

NUM_PLAYERS = 2
P0 = 0  # Wolf
P1 = 1  # Raven

# Unit Types: 0-3 are colored units, 4 is Royal
UNIT_RED = 0
UNIT_GREEN = 1
UNIT_BLUE = 2
UNIT_YELLOW = 3
UNIT_ROYAL = 4
NUM_UNIT_TYPES = 5
NUM_COLORED_UNITS = 4

# Coin distribution per game rules
COINS_PER_COLOR = 5
INITIAL_BAG_PER_COLOR = 2
INITIAL_SUPPLY_PER_COLOR = 3  # 5 - 2 = 3

# Game end
MAX_CONTROL_MARKERS = 6
MAX_STEP_COUNT = 300

# =============================================================================
# HEX GRID CONSTANTS (Flat-topped, Radius 3, 37 hexes)
# =============================================================================
# 
# Using AXIAL COORDINATES as specified in the rules:
#   - Center of board is (0, 0)
#   - q-axis points East, r-axis points Southeast
#   - Valid hexes satisfy: max(|q|, |r|, |-q-r|) <= 3
#
# Control point locations from rules (line 137):
#   - Wolf starting: (-1, 3), (2, 1)
#   - Raven starting: (-2, -1), (1, -3)
#   - Neutral: (-3, 1), (-2, 2), (-1, 0), (1, 0), (2, -2), (3, -1)

NUM_HEXES = 37

def _generate_hex_coords():
    """Generate (q, r) axial coordinates for all 37 hexes.
    
    Uses centered axial coordinates matching the rules specification.
    The board is a radius-3 hex grid where valid hexes satisfy:
    max(|q|, |r|, |s|) <= 3  where s = -q - r
    
    Returns coordinates sorted by (q, r) for consistent indexing.
    """
    coords = []
    for q in range(-3, 4):  # q from -3 to 3
        for r in range(-3, 4):  # r from -3 to 3
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= 3:
                coords.append((q, r))
    # Sort by column (q) then row (r) for consistent indexing
    coords.sort(key=lambda c: (c[0], c[1]))
    return jnp.array(coords, dtype=jnp.int32)

HEX_COORDS = _generate_hex_coords()

# Neighbor directions for axial coordinates (flat-topped hex)
# 6 directions: E, NE, NW, W, SW, SE
HEX_DELTAS = jnp.array([[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]], dtype=jnp.int32)

def _generate_neighbor_map():
    """Generates adjacency list for the 37-hex board."""
    neighbor_map = -jnp.ones((NUM_HEXES, 6), dtype=jnp.int32)
    
    for i in range(NUM_HEXES):
        curr_q, curr_r = int(HEX_COORDS[i, 0]), int(HEX_COORDS[i, 1])
        for d in range(6):
            nq = curr_q + int(HEX_DELTAS[d, 0])
            nr = curr_r + int(HEX_DELTAS[d, 1])
            # Find if neighbor exists
            for j in range(NUM_HEXES):
                if int(HEX_COORDS[j, 0]) == nq and int(HEX_COORDS[j, 1]) == nr:
                    neighbor_map = neighbor_map.at[i, d].set(j)
                    break
    return neighbor_map

NEIGHBOR_MAP = _generate_neighbor_map()

# =============================================================================
# CONTROL POINT LOCATIONS (using axial coordinates from rules)
# =============================================================================

NUM_CONTROL_POINTS = 10

def _find_hex_idx(q, r):
    """Find hex index for given axial coordinates (q, r)."""
    for i in range(NUM_HEXES):
        if int(HEX_COORDS[i, 0]) == q and int(HEX_COORDS[i, 1]) == r:
            return i
    return -1

# Control point locations from rules (line 137) in axial coordinates:
# Wolf (P0) starting positions: (-1, 3), (2, 1)
_WOLF_START_COORDS = [(-1, 3), (2, 1)]
LOCS_P0_START = jnp.array([_find_hex_idx(q, r) for q, r in _WOLF_START_COORDS], dtype=jnp.int32)

# Raven (P1) starting positions: (-2, -1), (1, -3)
_RAVEN_START_COORDS = [(-2, -1), (1, -3)]
LOCS_P1_START = jnp.array([_find_hex_idx(q, r) for q, r in _RAVEN_START_COORDS], dtype=jnp.int32)

# Neutral control points: (-3, 1), (-2, 2), (-1, 0), (1, 0), (2, -2), (3, -1)
_NEUTRAL_COORDS = [(-3, 1), (-2, 2), (-1, 0), (1, 0), (2, -2), (3, -1)]
_NEUTRAL_INDICES = [_find_hex_idx(q, r) for q, r in _NEUTRAL_COORDS]

# All control points (10 total: 2 wolf + 2 raven + 6 neutral)
_ALL_CONTROL_COORDS = _WOLF_START_COORDS + _RAVEN_START_COORDS + _NEUTRAL_COORDS
CONTROL_POINT_HEXES = jnp.array([_find_hex_idx(q, r) for q, r in _ALL_CONTROL_COORDS], dtype=jnp.int32)

# Boolean mask for control points
IS_CONTROL_POINT = jnp.zeros(NUM_HEXES, dtype=jnp.bool_).at[CONTROL_POINT_HEXES].set(True)

# =============================================================================
# ACTION SPACE (659 actions - Hex-based approach)
# =============================================================================

# Action encoding:
# 0-147:   Deploy/Bolster at Hex × Unit Type (37 hexes × 4 colored units = 148)
# 148-369: Move from Hex (37 × 6 directions = 222)
# 370-591: Attack from Hex (37 × 6 directions = 222)
# 592-628: Control at Hex (37)
# 629-648: Recruit Target × Discard Coin (4 targets × 5 coin types = 20)
# 649-653: Claim Initiative with Discard Coin (5 coin types)
# 654-658: Pass with Discard Coin (5 coin types)

ACT_DEPLOY_START = 0
ACT_DEPLOY_END = 148
ACT_MOVE_START = 148
ACT_MOVE_END = 370
ACT_ATTACK_START = 370
ACT_ATTACK_END = 592
ACT_CONTROL_START = 592
ACT_CONTROL_END = 629
ACT_RECRUIT_START = 629
ACT_RECRUIT_END = 649
ACT_INITIATIVE_START = 649
ACT_INITIATIVE_END = 654
ACT_PASS_START = 654
ACT_PASS_END = 659
NUM_ACTIONS = 659


def _decode_action(action: Array):
    """Decode action into type and parameters.
    
    Returns:
        action_type: 0=deploy, 1=move, 2=attack, 3=control, 4=recruit, 5=initiative, 6=pass
        hex_idx: Target/source hex for deploy/move/attack/control
        direction: Direction for move/attack (0-5)
        recruit_unit: Target unit for recruit (0-3)
        discard_coin: Coin type to discard for face-down actions (0-4)
        deploy_unit: Unit type for deploy/bolster (0-3, colored only)
    """
    is_deploy = (action >= ACT_DEPLOY_START) & (action < ACT_DEPLOY_END)
    is_move = (action >= ACT_MOVE_START) & (action < ACT_MOVE_END)
    is_attack = (action >= ACT_ATTACK_START) & (action < ACT_ATTACK_END)
    is_control = (action >= ACT_CONTROL_START) & (action < ACT_CONTROL_END)
    is_recruit = (action >= ACT_RECRUIT_START) & (action < ACT_RECRUIT_END)
    is_initiative = (action >= ACT_INITIATIVE_START) & (action < ACT_INITIATIVE_END)
    is_pass = (action >= ACT_PASS_START) & (action < ACT_PASS_END)
    
    action_type = jnp.where(is_deploy, 0,
                  jnp.where(is_move, 1,
                  jnp.where(is_attack, 2,
                  jnp.where(is_control, 3,
                  jnp.where(is_recruit, 4,
                  jnp.where(is_initiative, 5, 6))))))
    
    # Deploy: action = ACT_DEPLOY_START + hex_idx * NUM_COLORED_UNITS + unit_type
    deploy_unit = jnp.where(is_deploy, (action - ACT_DEPLOY_START) % NUM_COLORED_UNITS, -1)
    
    # Decode hex_idx and direction based on action type
    hex_idx = jnp.where(is_deploy, (action - ACT_DEPLOY_START) // NUM_COLORED_UNITS,
              jnp.where(is_move, (action - ACT_MOVE_START) // 6,
              jnp.where(is_attack, (action - ACT_ATTACK_START) // 6,
              jnp.where(is_control, action - ACT_CONTROL_START, -1))))
    
    direction = jnp.where(is_move, (action - ACT_MOVE_START) % 6,
                jnp.where(is_attack, (action - ACT_ATTACK_START) % 6, -1))
    
    # Recruit: action = ACT_RECRUIT_START + recruit_target * NUM_UNIT_TYPES + discard_coin
    recruit_unit = jnp.where(is_recruit, (action - ACT_RECRUIT_START) // NUM_UNIT_TYPES, -1)
    
    # Discard coin for face-down actions (recruit, initiative, pass)
    discard_coin = jnp.where(is_recruit, (action - ACT_RECRUIT_START) % NUM_UNIT_TYPES,
                   jnp.where(is_initiative, action - ACT_INITIATIVE_START,
                   jnp.where(is_pass, action - ACT_PASS_START, -1)))
    
    return (action_type.astype(jnp.int32), hex_idx.astype(jnp.int32),
            direction.astype(jnp.int32), recruit_unit.astype(jnp.int32),
            discard_coin.astype(jnp.int32), deploy_unit.astype(jnp.int32))


# =============================================================================
# OBSERVATION CONSTANTS
# =============================================================================

# Observation shape: (NUM_HEXES, num_channels) + global features
# Per-hex channels (Royal excluded — never on board):
#   0-3: My units (count per colored type, 4 types: R/G/B/Y)
#   4-7: Opponent units (count per colored type, 4 types)
#   8:  My control marker
#   9:  Opponent control marker
#   10: Is control point
NUM_HEX_CHANNELS = 11

# Global features (broadcast as constant planes in 2D grid):
#   - My hand (5 values, histogram)
#   - Opponent's face-up discard (5 values, public)
#   - My supply (5 values)
#   - Opponent supply (5 values)
#   - My cemetery (5 values)
#   - Opponent cemetery (5 values)
#   - My face-up discard (5 values, player knows own)
#   - My face-down discard (5 values, player knows own contents)
#   - My bag composition (5 values, player can track own bag)
#   - Initiative holder (1 value: 0 or 1)
#   - Is my turn (1 value)
#   - Initiative taken this round (1 value)
#   - Opponent face-down discard count (1 value, public count only)
#   - Opponent bag coin count (1 value, public)
#   - Opponent hand coin count (1 value, public count only)
#   - My hand coin count (1 value)
NUM_GLOBAL_FEATURES = 5 * 9 + 3 + 4

# 2D grid for Conv2D compatibility
# Map 37 hexes onto a 7×7 grid via (q+3, r+3). 12 cells are padding (zero).
GRID_SIZE = 7
GRID_CHANNELS = NUM_HEX_CHANNELS + NUM_GLOBAL_FEATURES  # 11 + 52 = 63
OBS_SHAPE = (GRID_SIZE, GRID_SIZE, GRID_CHANNELS)  # (7, 7, 63)
OBS_SIZE = GRID_SIZE * GRID_SIZE * GRID_CHANNELS  # kept for reference

# Precomputed hex-to-grid mapping: hex i -> grid position (row, col) = (q+3, r+3)
HEX_TO_GRID_ROW = HEX_COORDS[:, 0] + 3  # q + 3
HEX_TO_GRID_COL = HEX_COORDS[:, 1] + 3  # r + 3

# Valid hex mask on the 7×7 grid (True at positions with hexes, False at padding)
def _build_valid_hex_mask():
    mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)
    for i in range(NUM_HEXES):
        r = int(HEX_TO_GRID_ROW[i])
        c = int(HEX_TO_GRID_COL[i])
        mask = mask.at[r, c].set(True)
    return mask

VALID_HEX_MASK = _build_valid_hex_mask()  # (7, 7) bool


# =============================================================================
# GAME STATE (Using histogram storage for efficiency)
# =============================================================================

class GameState(NamedTuple):
    """JAX-compatible game state as NamedTuple."""
    current_player: Array = jnp.int32(0)
    phase: Array = jnp.int32(0)  # 0: Draw, 1: Action
    
    # Board state
    board_unit_type: Array = -jnp.ones(NUM_HEXES, dtype=jnp.int32)
    board_owner: Array = -jnp.ones(NUM_HEXES, dtype=jnp.int32)
    board_count: Array = jnp.zeros(NUM_HEXES, dtype=jnp.int32)
    control_owner: Array = -jnp.ones(NUM_HEXES, dtype=jnp.int32)
    
    # Player resources [2 players, 5 unit types] - histogram storage
    supply: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    bag: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    hand: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    discard_faceup: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    discard_facedown: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    cemetery: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    
    # Initiative
    initiative_player: Array = jnp.int32(0)
    initiative_taken_this_round: Array = FALSE
    
    # Turn tracking
    turn_in_round: Array = jnp.int32(0)
    
    # Game end
    terminated: Array = FALSE
    winner: Array = jnp.int32(-1)


# =============================================================================
# GAME LOGIC
# =============================================================================

class Game:
    """War Chest Simplified game logic."""
    
    def init(self, key: Array) -> GameState:
        """Initialize a new game state."""
        key, subkey = jax.random.split(key)
        
        # Random starting player
        start_player = jax.random.bernoulli(subkey).astype(jnp.int32)
        
        # Initial bag: 2 of each color + 1 Royal
        start_bag = jnp.array([INITIAL_BAG_PER_COLOR] * NUM_COLORED_UNITS + [1], dtype=jnp.int32)
        bag = jnp.stack([start_bag, start_bag])
        
        # Initial supply: 3 of each color, 0 Royal
        start_supply = jnp.array([INITIAL_SUPPLY_PER_COLOR] * NUM_COLORED_UNITS + [0], dtype=jnp.int32)
        supply = jnp.stack([start_supply, start_supply])
        
        # Setup control markers on starting positions
        control = -jnp.ones(NUM_HEXES, dtype=jnp.int32)
        control = control.at[LOCS_P0_START].set(P0)
        control = control.at[LOCS_P1_START].set(P1)
        
        state = GameState(
            current_player=start_player,
            initiative_player=start_player,
            phase=jnp.int32(0),  # Draw phase
            control_owner=control,
            bag=bag,
            supply=supply,
        )
        
        # Execute draw phase
        state = self._draw_phase(state, key)
        
        return state
    
    def _draw_phase(self, state: GameState, key: Array) -> GameState:
        """Execute the draw phase - each player draws 3 coins."""
        key, key1, key2 = jax.random.split(key, 3)
        
        # Draw for both players
        state = self._draw_for_player(state, P0, key1)
        state = self._draw_for_player(state, P1, key2)
        
        # Transition to action phase
        state = state._replace(
            phase=jnp.int32(1),
            turn_in_round=jnp.int32(0),
            current_player=state.initiative_player,
            initiative_taken_this_round=FALSE,
        )
        
        return state
    
    def _draw_for_player(self, state: GameState, player: int, key: Array) -> GameState:
        """Draw up to 3 coins for a player from their bag.
        
        Per rules: draw from bag one at a time. If bag becomes empty
        mid-draw, shuffle all discard (face-up + face-down) into bag
        and continue drawing. If still no coins, stop.
        """
        bag = state.bag[player]
        hand = state.hand[player]
        discard_up = state.discard_faceup[player]
        discard_down = state.discard_facedown[player]
        
        # Draw 3 coins one at a time with mid-draw refill
        def draw_one(carry, _):
            _bag, _hand, _discard_up, _discard_down, _key = carry
            _key, subkey = jax.random.split(_key)
            
            # Mid-draw refill: if bag is empty, shuffle discards into bag
            bag_empty = jnp.sum(_bag) == 0
            _bag = jnp.where(bag_empty, _bag + _discard_up + _discard_down, _bag)
            _discard_up = jnp.where(bag_empty, jnp.zeros_like(_discard_up), _discard_up)
            _discard_down = jnp.where(bag_empty, jnp.zeros_like(_discard_down), _discard_down)
            
            # Create probability distribution from (possibly refilled) bag
            bag_total = jnp.sum(_bag)
            has_coins = bag_total > 0
            
            # Normalize probabilities (avoid division by zero)
            probs = _bag.astype(jnp.float32)
            probs = probs / jnp.maximum(bag_total, 1)
            probs = jnp.where(has_coins, probs, jnp.ones(NUM_UNIT_TYPES) / NUM_UNIT_TYPES)
            
            # Draw a coin
            drawn_type = jax.random.choice(subkey, NUM_UNIT_TYPES, p=probs)
            
            # Update bag and hand only if we actually have coins
            new_bag = jnp.where(has_coins, _bag.at[drawn_type].add(-1), _bag)
            new_hand = jnp.where(has_coins, _hand.at[drawn_type].add(1), _hand)
            
            return (new_bag, new_hand, _discard_up, _discard_down, _key), None
        
        # Draw 3 times (will draw fewer if not enough coins exist)
        (bag, hand, discard_up, discard_down, _), _ = lax.scan(
            draw_one, (bag, hand, discard_up, discard_down, key), None, length=3
        )
        
        # Update state
        new_bag = state.bag.at[player].set(bag)
        new_hand = state.hand.at[player].set(hand)
        new_discard_up = state.discard_faceup.at[player].set(discard_up)
        new_discard_down = state.discard_facedown.at[player].set(discard_down)
        
        return state._replace(
            bag=new_bag,
            hand=new_hand,
            discard_faceup=new_discard_up,
            discard_facedown=new_discard_down,
        )
    
    def step(self, state: GameState, action: Array, key: Array) -> GameState:
        """Execute one action."""
        # Decode action (includes deploy_unit and discard_coin)
        action_type, hex_idx, direction, recruit_unit, discard_coin, deploy_unit = _decode_action(action)
        
        # Find which unit type to consume from hand:
        # - For deploy actions: unit type encoded in action (deploy_unit)
        # - For face-down discard actions (recruit/initiative/pass): coin encoded in action
        # - For face-up board actions (move/attack/control): determined by board context
        unit_type = self._get_required_unit_for_action(state, action, discard_coin, deploy_unit)
        
        # Execute action based on type
        state = lax.switch(
            action_type,
            [
                lambda s: self._do_deploy(s, hex_idx, unit_type),
                lambda s: self._do_move(s, hex_idx, direction, unit_type),
                lambda s: self._do_attack(s, hex_idx, direction, unit_type),
                lambda s: self._do_control(s, hex_idx, unit_type),
                lambda s: self._do_recruit(s, recruit_unit, unit_type),
                lambda s: self._do_initiative(s, unit_type),
                lambda s: self._do_pass(s, unit_type),
            ],
            state
        )
        
        # Check for win condition
        state = self._check_victory(state)
        
        # Handle turn/round transitions
        state = lax.cond(
            state.terminated,
            lambda: state,
            lambda: self._advance_turn(state, key)
        )
        
        return state
    
    def _get_required_unit_for_action(self, state: GameState, action: Array, discard_coin: Array, deploy_unit: Array) -> Array:
        """Find a unit type in hand that can execute this action.
        
        For deploy actions: unit type is explicitly encoded via deploy_unit.
        For face-down discard actions (recruit/initiative/pass): coin is
        explicitly encoded via discard_coin.
        For face-up board actions (move/attack/control): determined by
        board context. Prefers colored units (0-3) over Royal.
        """
        action_type, _, _, _, _, _ = _decode_action(action)
        is_deploy = action_type == 0
        is_facedown = action_type >= 4  # recruit=4, initiative=5, pass=6
        
        p = state.current_player
        hand = state.hand[p]
        
        def check_unit(u):
            in_hand = hand[u] > 0
            valid = self._is_action_valid_for_unit(state, action, u)
            return jnp.where(in_hand & valid, u, jnp.int32(-1))
        
        # Check all unit types for board-context actions (move/attack/control)
        results = jax.vmap(check_unit)(jnp.arange(NUM_UNIT_TYPES))
        colored_results = results[:NUM_COLORED_UNITS]
        has_colored = jnp.any(colored_results >= 0)
        colored_pick = jnp.where(colored_results >= 0, colored_results, jnp.int32(NUM_UNIT_TYPES))
        best_colored = jnp.min(colored_pick)
        board_unit = jnp.where(has_colored, best_colored, results[UNIT_ROYAL])
        
        # Deploy: unit type from action; face-down: discard coin; otherwise: board context
        return jnp.where(is_deploy, deploy_unit,
               jnp.where(is_facedown, discard_coin, board_unit))
    
    def _is_action_valid_for_unit(self, state: GameState, action: Array, unit_type: Array) -> Array:
        """Check if a specific unit type can execute the given action."""
        action_type, hex_idx, direction, recruit_unit, discard_coin, deploy_unit = _decode_action(action)
        p = state.current_player
        opp = 1 - p
        
        def check_deploy():
            is_empty = state.board_unit_type[hex_idx] == -1
            is_controlled = state.control_owner[hex_idx] == p
            is_mine = (state.board_owner[hex_idx] == p) & (state.board_unit_type[hex_idx] == unit_type)
            is_royal = unit_type == UNIT_ROYAL
            # The unit type must match the deploy_unit encoded in the action
            correct_unit = unit_type == deploy_unit
            
            # Cannot deploy if unit is already on board elsewhere (unless bolstering)
            unit_on_board = jnp.any((state.board_owner == p) & (state.board_unit_type == unit_type))
            
            # Deploy: empty + controlled + not royal + not already on board + correct unit
            # Bolster: own unit already there + correct unit
            return ((is_empty & is_controlled & ~is_royal & ~unit_on_board) | is_mine) & correct_unit
        
        def check_move():
            src_hex = hex_idx
            board_match = (state.board_owner[src_hex] == p) & (state.board_unit_type[src_hex] == unit_type)
            dst_hex = NEIGHBOR_MAP[src_hex, direction]
            valid_neighbor = dst_hex >= 0
            dst_empty = lax.cond(valid_neighbor, lambda: state.board_unit_type[dst_hex] == -1, lambda: FALSE)
            return board_match & valid_neighbor & dst_empty
        
        def check_attack():
            src_hex = hex_idx
            board_match = (state.board_owner[src_hex] == p) & (state.board_unit_type[src_hex] == unit_type)
            dst_hex = NEIGHBOR_MAP[src_hex, direction]
            valid_neighbor = dst_hex >= 0
            has_enemy = lax.cond(valid_neighbor, lambda: state.board_owner[dst_hex] == opp, lambda: FALSE)
            return board_match & valid_neighbor & has_enemy
        
        def check_control():
            board_match = (state.board_owner[hex_idx] == p) & (state.board_unit_type[hex_idx] == unit_type)
            can_flip = state.control_owner[hex_idx] != p
            is_control_point = IS_CONTROL_POINT[hex_idx]
            return board_match & can_flip & is_control_point
        
        def check_recruit():
            has_supply = state.supply[p, recruit_unit] > 0
            not_royal_recruit = recruit_unit != UNIT_ROYAL
            # The discard coin must match this unit_type
            correct_coin = unit_type == discard_coin
            return has_supply & not_royal_recruit & correct_coin
        
        def check_initiative():
            not_holder = state.initiative_player != p
            not_taken = ~state.initiative_taken_this_round
            correct_coin = unit_type == discard_coin
            return not_holder & not_taken & correct_coin
        
        def check_pass():
            correct_coin = unit_type == discard_coin
            return correct_coin
        
        result = lax.switch(
            action_type,
            [check_deploy, check_move, check_attack, check_control, check_recruit, check_initiative, check_pass]
        )
        
        return result
    
    def _consume_coin(self, state: GameState, unit_type: Array) -> GameState:
        """Remove one coin of given type from current player's hand."""
        p = state.current_player
        new_hand = state.hand.at[p, unit_type].add(-1)
        return state._replace(hand=new_hand)
    
    def _add_to_discard(self, state: GameState, unit_type: Array, face_up: Array) -> GameState:
        """Add a coin to current player's discard pile."""
        p = state.current_player
        new_discard_up = jnp.where(face_up, state.discard_faceup.at[p, unit_type].add(1), state.discard_faceup)
        new_discard_down = jnp.where(face_up, state.discard_facedown, state.discard_facedown.at[p, unit_type].add(1))
        return state._replace(discard_faceup=new_discard_up, discard_facedown=new_discard_down)
    
    def _do_deploy(self, state: GameState, hex_idx: Array, unit_type: Array) -> GameState:
        """Deploy a new unit or bolster existing unit."""
        p = state.current_player
        
        # Check if bolstering (unit already there)
        is_bolster = (state.board_unit_type[hex_idx] == unit_type) & (state.board_owner[hex_idx] == p)
        
        # Consume coin from hand (goes onto the board, not discarded)
        state = self._consume_coin(state, unit_type)
        
        # Update board
        new_count = state.board_count[hex_idx] + 1
        new_board_unit_type = state.board_unit_type.at[hex_idx].set(unit_type)
        new_board_owner = state.board_owner.at[hex_idx].set(p)
        new_board_count = state.board_count.at[hex_idx].set(new_count)
        
        return state._replace(
            board_unit_type=new_board_unit_type,
            board_owner=new_board_owner,
            board_count=new_board_count,
        )
    
    def _do_move(self, state: GameState, src_hex: Array, direction: Array, unit_type: Array) -> GameState:
        """Move a unit to adjacent hex."""
        p = state.current_player
        dst_hex = NEIGHBOR_MAP[src_hex, direction]
        
        # Consume coin and add to face-up discard (public action)
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=TRUE)
        
        # Move unit from src to dst
        u = state.board_unit_type[src_hex]
        o = state.board_owner[src_hex]
        c = state.board_count[src_hex]
        
        new_board_unit_type = state.board_unit_type.at[src_hex].set(-1).at[dst_hex].set(u)
        new_board_owner = state.board_owner.at[src_hex].set(-1).at[dst_hex].set(o)
        new_board_count = state.board_count.at[src_hex].set(0).at[dst_hex].set(c)
        
        return state._replace(
            board_unit_type=new_board_unit_type,
            board_owner=new_board_owner,
            board_count=new_board_count,
        )
    
    def _do_attack(self, state: GameState, src_hex: Array, direction: Array, unit_type: Array) -> GameState:
        """Attack adjacent enemy unit."""
        p = state.current_player
        opp = 1 - p
        dst_hex = NEIGHBOR_MAP[src_hex, direction]
        
        # Consume coin and add to face-up discard (public action)
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=TRUE)
        
        # Damage enemy unit
        enemy_unit_type = state.board_unit_type[dst_hex]
        new_count = state.board_count[dst_hex] - 1
        is_dead = new_count <= 0
        
        # Add to enemy's cemetery (coin is removed from game)
        new_cemetery = state.cemetery.at[opp, enemy_unit_type].add(1)
        
        # Update board
        new_board_count = state.board_count.at[dst_hex].set(jnp.maximum(0, new_count))
        new_board_unit_type = lax.cond(
            is_dead,
            lambda: state.board_unit_type.at[dst_hex].set(-1),
            lambda: state.board_unit_type
        )
        new_board_owner = lax.cond(
            is_dead,
            lambda: state.board_owner.at[dst_hex].set(-1),
            lambda: state.board_owner
        )
        
        return state._replace(
            board_unit_type=new_board_unit_type,
            board_owner=new_board_owner,
            board_count=new_board_count,
            cemetery=new_cemetery,
        )
    
    def _do_control(self, state: GameState, hex_idx: Array, unit_type: Array) -> GameState:
        """Claim control of the location."""
        p = state.current_player
        
        # Consume coin and add to face-up discard (public action)
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=TRUE)
        
        # Update control
        new_control = state.control_owner.at[hex_idx].set(p)
        
        return state._replace(control_owner=new_control)
    
    def _do_recruit(self, state: GameState, target_unit: Array, unit_type: Array) -> GameState:
        """Recruit a new coin from supply."""
        p = state.current_player
        
        # Pay cost: discard coin face-down (hidden action)
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=FALSE)
        
        # Take from supply
        new_supply = state.supply.at[p, target_unit].add(-1)
        
        # Add recruited coin to face-up discard (it's a gain, visible)
        new_discard_up = state.discard_faceup.at[p, target_unit].add(1)  # recruited coin is public
        
        return state._replace(
            supply=new_supply,
            discard_faceup=new_discard_up,
        )
    
    def _do_initiative(self, state: GameState, unit_type: Array) -> GameState:
        """Claim the initiative marker."""
        p = state.current_player
        
        # Pay cost: discard coin face-down (hidden action)
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=FALSE)
        
        return state._replace(
            initiative_player=p,
            initiative_taken_this_round=TRUE,
        )
    
    def _do_pass(self, state: GameState, unit_type: Array) -> GameState:
        """Pass by discarding a coin."""
        # Pay cost: discard coin face-down (hidden action)
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=FALSE)
        
        return state
    
    def _check_victory(self, state: GameState) -> GameState:
        """Check if a player has won (6 control points)."""
        p0_score = jnp.sum((state.control_owner == P0) & IS_CONTROL_POINT)
        p1_score = jnp.sum((state.control_owner == P1) & IS_CONTROL_POINT)
        
        p0_win = p0_score >= MAX_CONTROL_MARKERS
        p1_win = p1_score >= MAX_CONTROL_MARKERS
        
        terminated = p0_win | p1_win
        winner = jnp.where(p0_win, P0, jnp.where(p1_win, P1, jnp.int32(-1)))
        
        return state._replace(terminated=terminated, winner=winner)
    
    def _advance_turn(self, state: GameState, key: Array) -> GameState:
        """Advance to next player's turn or new round.
        
        Per rules: players alternate turns. If a player runs out of hand
        coins, the other continues. Round ends when both hands are empty.
        """
        p = state.current_player
        opp = 1 - p
        my_hand_empty = jnp.sum(state.hand[p]) == 0
        opp_hand_empty = jnp.sum(state.hand[opp]) == 0
        both_empty = my_hand_empty & opp_hand_empty
        
        def new_round():
            return self._draw_phase(state, key)
        
        def continue_play():
            # If opponent has coins, switch to them; otherwise stay on current player
            next_player = jnp.where(opp_hand_empty, p, opp)
            return state._replace(
                turn_in_round=state.turn_in_round + 1,
                current_player=next_player,
            )
        
        return lax.cond(both_empty, new_round, continue_play)
    
    def legal_action_mask(self, state: GameState) -> Array:
        """Compute legal action mask."""
        p = state.current_player
        hand = state.hand[p]
        has_any_coin = jnp.sum(hand) > 0
        
        def is_legal_single(action_idx):
            def check_unit(u):
                has_u = hand[u] > 0
                valid = self._is_action_valid_for_unit(state, action_idx, u)
                return has_u & valid
            
            return jax.vmap(check_unit)(jnp.arange(NUM_UNIT_TYPES)).any()
        
        mask = jax.vmap(is_legal_single)(jnp.arange(NUM_ACTIONS))
        return mask & has_any_coin
    
    def is_terminal(self, state: GameState) -> Array:
        """Check if game is over."""
        return state.terminated
    
    def rewards(self, state: GameState) -> Array:
        """Compute rewards for both players."""
        return lax.cond(
            state.winner >= 0,
            lambda: jnp.float32([-1.0, -1.0]).at[state.winner].set(1.0),
            lambda: jnp.zeros(2, dtype=jnp.float32)
        )
    
    def observe(self, state: GameState, player_id: Array) -> Array:
        """Generate observation for a player as a 2D grid.
        
        Includes all public info (per rules Section 5) plus own private info.
        The observation is a (7, 7, 63) array where:
          - Channels 0-10: per-hex board features mapped onto a 7×7 grid
          - Channels 11-62: global features broadcast as constant planes
        Invalid grid cells (padding) are all zeros.
        
        Args:
            state: Current game state
            player_id: Player to generate observation for (0 or 1)
            
        Returns:
            Observation array of shape (7, 7, 63)
        """
        opp = 1 - player_id
        
        # --- Per-hex board features (37 hexes × 11 channels) ---
        obs_board = jnp.zeros((NUM_HEXES, NUM_HEX_CHANNELS), dtype=jnp.float32)
        
        # My units (channels 0-3: count per colored unit type, Royal excluded)
        for ut in range(NUM_COLORED_UNITS):
            my_unit_mask = (state.board_owner == player_id) & (state.board_unit_type == ut)
            obs_board = obs_board.at[:, ut].set(
                jnp.where(my_unit_mask, state.board_count.astype(jnp.float32), 0.0)
            )
        
        # Opponent units (channels 4-7)
        for ut in range(NUM_COLORED_UNITS):
            opp_unit_mask = (state.board_owner == opp) & (state.board_unit_type == ut)
            obs_board = obs_board.at[:, NUM_COLORED_UNITS + ut].set(
                jnp.where(opp_unit_mask, state.board_count.astype(jnp.float32), 0.0)
            )
        
        # Control markers
        obs_board = obs_board.at[:, 8].set((state.control_owner == player_id).astype(jnp.float32))
        obs_board = obs_board.at[:, 9].set((state.control_owner == opp).astype(jnp.float32))
        obs_board = obs_board.at[:, 10].set(IS_CONTROL_POINT.astype(jnp.float32))
        
        # --- Place hex features onto 7×7 grid ---
        obs_grid = jnp.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=jnp.float32)
        
        # Scatter hex features to grid positions
        # obs_board is (37, 11), we place each hex's 11 channels at its grid (row, col)
        obs_grid = obs_grid.at[HEX_TO_GRID_ROW, HEX_TO_GRID_COL, :NUM_HEX_CHANNELS].set(obs_board)
        
        # --- Global features as broadcast planes ---
        my_hand = state.hand[player_id].astype(jnp.float32)                   # 5 (private)
        opp_visible_discard = state.discard_faceup[opp].astype(jnp.float32)   # 5 (public)
        my_supply = state.supply[player_id].astype(jnp.float32)               # 5 (public)
        opp_supply = state.supply[opp].astype(jnp.float32)                    # 5 (public)
        my_cemetery = state.cemetery[player_id].astype(jnp.float32)           # 5 (public)
        opp_cemetery = state.cemetery[opp].astype(jnp.float32)                # 5 (public)
        my_discard_up = state.discard_faceup[player_id].astype(jnp.float32)   # 5 (own info)
        my_discard_down = state.discard_facedown[player_id].astype(jnp.float32) # 5 (own info)
        my_bag = state.bag[player_id].astype(jnp.float32)                     # 5 (own trackable)
        
        is_my_turn = (state.current_player == player_id).astype(jnp.float32)
        i_have_initiative = (state.initiative_player == player_id).astype(jnp.float32)
        init_taken = state.initiative_taken_this_round.astype(jnp.float32)
        
        # Scalar counts (public info per rules Section 5)
        opp_facedown_count = jnp.sum(state.discard_facedown[opp]).astype(jnp.float32)
        opp_bag_count = jnp.sum(state.bag[opp]).astype(jnp.float32)
        opp_hand_count = jnp.sum(state.hand[opp]).astype(jnp.float32)
        my_hand_count = jnp.sum(state.hand[player_id]).astype(jnp.float32)
        
        global_features = jnp.concatenate([
            my_hand,                # 5
            opp_visible_discard,    # 5
            my_supply,              # 5
            opp_supply,             # 5
            my_cemetery,            # 5
            opp_cemetery,           # 5
            my_discard_up,          # 5
            my_discard_down,        # 5
            my_bag,                 # 5
            jnp.array([is_my_turn, i_have_initiative, init_taken]),  # 3
            jnp.array([opp_facedown_count, opp_bag_count,
                       opp_hand_count, my_hand_count]),  # 4
        ])  # shape: (52,)
        
        # Broadcast each global scalar as a constant plane over valid hexes only
        # global_features shape: (52,) -> (1, 1, 52) -> broadcast to (7, 7, 52)
        global_planes = jnp.broadcast_to(
            global_features[None, None, :], (GRID_SIZE, GRID_SIZE, NUM_GLOBAL_FEATURES)
        )
        # Mask out padding cells: only valid hex positions get global features
        global_planes = global_planes * VALID_HEX_MASK[:, :, None]
        
        obs_grid = obs_grid.at[:, :, NUM_HEX_CHANNELS:].set(global_planes)
        
        return obs_grid