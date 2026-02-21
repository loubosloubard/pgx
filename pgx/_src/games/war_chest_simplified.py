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

NUM_HEXES = 37

def _generate_hex_coords():
    """Generate (q, r) axial coordinates for all 37 hexes."""
    coords = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= 3:
                coords.append((q, r))
    coords.sort(key=lambda c: (c[0], c[1]))
    return jnp.array(coords, dtype=jnp.int32)

HEX_COORDS = _generate_hex_coords()

# 6 directions: E, NE, NW, W, SW, SE
HEX_DELTAS = jnp.array([[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]], dtype=jnp.int32)

def _generate_neighbor_map():
    """Generates adjacency list for the 37-hex board."""
    neighbor_map = -jnp.ones((NUM_HEXES, 6), dtype=jnp.int32)
    for i in range(NUM_HEXES):
        curr_q, curr_r = int(HEX_COORDS[i, 0]), int(HEX_COORDS[i, 1])
        for d in range(6):
            nq, nr = curr_q + int(HEX_DELTAS[d, 0]), curr_r + int(HEX_DELTAS[d, 1])
            for j in range(NUM_HEXES):
                if int(HEX_COORDS[j, 0]) == nq and int(HEX_COORDS[j, 1]) == nr:
                    neighbor_map = neighbor_map.at[i, d].set(j)
                    break
    return neighbor_map

NEIGHBOR_MAP = _generate_neighbor_map()

# =============================================================================
# CONTROL POINT LOCATIONS
# =============================================================================

NUM_CONTROL_POINTS = 10

def _find_hex_idx(q, r):
    for i in range(NUM_HEXES):
        if int(HEX_COORDS[i, 0]) == q and int(HEX_COORDS[i, 1]) == r:
            return i
    return -1

_WOLF_START_COORDS = [(-1, 3), (2, 1)]
LOCS_P0_START = jnp.array([_find_hex_idx(q, r) for q, r in _WOLF_START_COORDS], dtype=jnp.int32)

_RAVEN_START_COORDS = [(-2, -1), (1, -3)]
LOCS_P1_START = jnp.array([_find_hex_idx(q, r) for q, r in _RAVEN_START_COORDS], dtype=jnp.int32)

_NEUTRAL_COORDS = [(-3, 1), (-2, 2), (-1, 0), (1, 0), (2, -2), (3, -1)]
_NEUTRAL_INDICES = [_find_hex_idx(q, r) for q, r in _NEUTRAL_COORDS]

_ALL_CONTROL_COORDS = _WOLF_START_COORDS + _RAVEN_START_COORDS + _NEUTRAL_COORDS
CONTROL_POINT_HEXES = jnp.array([_find_hex_idx(q, r) for q, r in _ALL_CONTROL_COORDS], dtype=jnp.int32)

# Boolean mask for control points (37,)
IS_CONTROL_POINT = jnp.zeros(NUM_HEXES, dtype=jnp.bool_).at[CONTROL_POINT_HEXES].set(True)

# =============================================================================
# ACTION SPACE
# =============================================================================

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
    is_deploy = (action >= ACT_DEPLOY_START) & (action < ACT_DEPLOY_END)
    is_move = (action >= ACT_MOVE_START) & (action < ACT_MOVE_END)
    is_attack = (action >= ACT_ATTACK_START) & (action < ACT_ATTACK_END)
    is_control = (action >= ACT_CONTROL_START) & (action < ACT_CONTROL_END)
    is_recruit = (action >= ACT_RECRUIT_START) & (action < ACT_RECRUIT_END)
    is_initiative = (action >= ACT_INITIATIVE_START) & (action < ACT_INITIATIVE_END)
    is_pass = (action >= ACT_PASS_START) & (action < ACT_PASS_END)
    
    action_type = jnp.where(is_deploy, 0, jnp.where(is_move, 1, jnp.where(is_attack, 2,
                  jnp.where(is_control, 3, jnp.where(is_recruit, 4,
                  jnp.where(is_initiative, 5, 6))))))
    
    deploy_unit = jnp.where(is_deploy, (action - ACT_DEPLOY_START) % NUM_COLORED_UNITS, -1)
    
    hex_idx = jnp.where(is_deploy, (action - ACT_DEPLOY_START) // NUM_COLORED_UNITS,
              jnp.where(is_move, (action - ACT_MOVE_START) // 6,
              jnp.where(is_attack, (action - ACT_ATTACK_START) // 6,
              jnp.where(is_control, action - ACT_CONTROL_START, -1))))
    
    direction = jnp.where(is_move, (action - ACT_MOVE_START) % 6,
                jnp.where(is_attack, (action - ACT_ATTACK_START) % 6, -1))
    
    recruit_unit = jnp.where(is_recruit, (action - ACT_RECRUIT_START) // NUM_UNIT_TYPES, -1)
    
    discard_coin = jnp.where(is_recruit, (action - ACT_RECRUIT_START) % NUM_UNIT_TYPES,
                   jnp.where(is_initiative, action - ACT_INITIATIVE_START,
                   jnp.where(is_pass, action - ACT_PASS_START, -1)))
    
    return (action_type.astype(jnp.int32), hex_idx.astype(jnp.int32),
            direction.astype(jnp.int32), recruit_unit.astype(jnp.int32),
            discard_coin.astype(jnp.int32), deploy_unit.astype(jnp.int32))

# =============================================================================
# OBSERVATION CONSTANTS
# =============================================================================

NUM_HEX_CHANNELS = 11
NUM_GLOBAL_FEATURES = 5 * 9 + 3 + 4
GRID_SIZE = 7
GRID_CHANNELS = NUM_HEX_CHANNELS + NUM_GLOBAL_FEATURES  # 63
OBS_SHAPE = (GRID_SIZE, GRID_SIZE, GRID_CHANNELS)
OBS_SIZE = GRID_SIZE * GRID_SIZE * GRID_CHANNELS

HEX_TO_GRID_ROW = HEX_COORDS[:, 0] + 3
HEX_TO_GRID_COL = HEX_COORDS[:, 1] + 3

def _build_valid_hex_mask():
    mask = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=jnp.bool_)
    for i in range(NUM_HEXES):
        mask = mask.at[int(HEX_TO_GRID_ROW[i]), int(HEX_TO_GRID_COL[i])].set(True)
    return mask

VALID_HEX_MASK = _build_valid_hex_mask()

# =============================================================================
# GAME STATE
# =============================================================================

class GameState(NamedTuple):
    current_player: Array = jnp.int32(0)
    phase: Array = jnp.int32(0)  # 0: Draw, 1: Action
    
    # Board state
    board_unit_type: Array = -jnp.ones(NUM_HEXES, dtype=jnp.int32)
    board_owner: Array = -jnp.ones(NUM_HEXES, dtype=jnp.int32)
    board_count: Array = jnp.zeros(NUM_HEXES, dtype=jnp.int32)
    control_owner: Array = -jnp.ones(NUM_HEXES, dtype=jnp.int32)
    
    # Player resources [2 players, 5 unit types]
    supply: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    bag: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    hand: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    discard_faceup: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    discard_facedown: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    cemetery: Array = jnp.zeros((2, NUM_UNIT_TYPES), dtype=jnp.int32)
    
    # Initiative & Trackers
    initiative_player: Array = jnp.int32(0)
    initiative_taken_this_round: Array = FALSE
    turn_in_round: Array = jnp.int32(0)
    
    terminated: Array = FALSE
    winner: Array = jnp.int32(-1)

# =============================================================================
# GAME LOGIC
# =============================================================================

class Game:
    def init(self, key: Array) -> GameState:
        key, subkey = jax.random.split(key)
        start_player = jax.random.bernoulli(subkey).astype(jnp.int32)
        
        start_bag = jnp.array([INITIAL_BAG_PER_COLOR] * NUM_COLORED_UNITS + [1], dtype=jnp.int32)
        bag = jnp.stack([start_bag, start_bag])
        
        start_supply = jnp.array([INITIAL_SUPPLY_PER_COLOR] * NUM_COLORED_UNITS + [0], dtype=jnp.int32)
        supply = jnp.stack([start_supply, start_supply])
        
        control = -jnp.ones(NUM_HEXES, dtype=jnp.int32)
        control = control.at[LOCS_P0_START].set(P0)
        control = control.at[LOCS_P1_START].set(P1)
        
        state = GameState(
            current_player=start_player,
            initiative_player=start_player,
            phase=jnp.int32(0),
            control_owner=control,
            bag=bag,
            supply=supply,
        )
        
        return self._draw_phase(state, key)
    
    def _draw_phase(self, state: GameState, key: Array) -> GameState:
        key, key1, key2 = jax.random.split(key, 3)
        state = self._draw_for_player(state, P0, key1)
        state = self._draw_for_player(state, P1, key2)
        return state._replace(
            phase=jnp.int32(1),
            turn_in_round=jnp.int32(0),
            current_player=state.initiative_player,
            initiative_taken_this_round=FALSE,
        )
    
    def _draw_for_player(self, state: GameState, player: int, key: Array) -> GameState:
        bag, hand = state.bag[player], state.hand[player]
        discard_up, discard_down = state.discard_faceup[player], state.discard_facedown[player]
        
        def draw_one(carry, _):
            _bag, _hand, _discard_up, _discard_down, _key = carry
            _key, subkey = jax.random.split(_key)
            
            bag_empty = jnp.sum(_bag) == 0
            _bag = jnp.where(bag_empty, _bag + _discard_up + _discard_down, _bag)
            _discard_up = jnp.where(bag_empty, jnp.zeros_like(_discard_up), _discard_up)
            _discard_down = jnp.where(bag_empty, jnp.zeros_like(_discard_down), _discard_down)
            
            bag_total = jnp.sum(_bag)
            has_coins = bag_total > 0
            
            probs = _bag.astype(jnp.float32) / jnp.maximum(bag_total, 1)
            probs = jnp.where(has_coins, probs, jnp.ones(NUM_UNIT_TYPES) / NUM_UNIT_TYPES)
            
            drawn_type = jax.random.choice(subkey, NUM_UNIT_TYPES, p=probs)
            new_bag = jnp.where(has_coins, _bag.at[drawn_type].add(-1), _bag)
            new_hand = jnp.where(has_coins, _hand.at[drawn_type].add(1), _hand)
            
            return (new_bag, new_hand, _discard_up, _discard_down, _key), None
        
        (bag, hand, discard_up, discard_down, _), _ = lax.scan(
            draw_one, (bag, hand, discard_up, discard_down, key), None, length=3
        )
        
        return state._replace(
            bag=state.bag.at[player].set(bag),
            hand=state.hand.at[player].set(hand),
            discard_faceup=state.discard_faceup.at[player].set(discard_up),
            discard_facedown=state.discard_facedown.at[player].set(discard_down),
        )

    def _get_required_unit_for_action(self, state: GameState, action: Array, discard_coin: Array, deploy_unit: Array, hex_idx: Array) -> Array:
        """Find a unit type in hand that can execute this action (O(1) complexity)."""
        action_type = jax.lax.select(action < ACT_DEPLOY_END, 0,
                      jax.lax.select(action < ACT_CONTROL_END, 1, 2))  # 0=Deploy, 1=Board(Move/Atk/Ctrl), 2=FaceDown
        
        # For board actions (Move/Attack/Control), we just consume the unit matching what's on the hex!
        board_unit = state.board_unit_type[hex_idx]
        
        return jnp.where(action_type == 0, deploy_unit,
               jnp.where(action_type == 2, discard_coin, board_unit))

    def step(self, state: GameState, action: Array, key: Array) -> GameState:
        action_type, hex_idx, direction, recruit_unit, discard_coin, deploy_unit = _decode_action(action)
        
        # O(1) fetch of consumed coin
        unit_type = self._get_required_unit_for_action(state, action, discard_coin, deploy_unit, hex_idx)
        
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
        
        state = self._check_victory(state)
        
        return lax.cond(state.terminated, lambda: state, lambda: self._advance_turn(state, key))
    
    def _consume_coin(self, state: GameState, unit_type: Array) -> GameState:
        p = state.current_player
        return state._replace(hand=state.hand.at[p, unit_type].add(-1))
    
    def _add_to_discard(self, state: GameState, unit_type: Array, face_up: Array) -> GameState:
        p = state.current_player
        new_discard_up = jnp.where(face_up, state.discard_faceup.at[p, unit_type].add(1), state.discard_faceup)
        new_discard_down = jnp.where(face_up, state.discard_facedown, state.discard_facedown.at[p, unit_type].add(1))
        return state._replace(discard_faceup=new_discard_up, discard_facedown=new_discard_down)
    
    def _do_deploy(self, state: GameState, hex_idx: Array, unit_type: Array) -> GameState:
        p = state.current_player
        state = self._consume_coin(state, unit_type)
        return state._replace(
            board_unit_type=state.board_unit_type.at[hex_idx].set(unit_type),
            board_owner=state.board_owner.at[hex_idx].set(p),
            board_count=state.board_count.at[hex_idx].add(1),
        )
    
    def _do_move(self, state: GameState, src_hex: Array, direction: Array, unit_type: Array) -> GameState:
        p = state.current_player
        dst_hex = NEIGHBOR_MAP[src_hex, direction]
        
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=TRUE)
        
        u, o, c = state.board_unit_type[src_hex], state.board_owner[src_hex], state.board_count[src_hex]
        return state._replace(
            board_unit_type=state.board_unit_type.at[src_hex].set(-1).at[dst_hex].set(u),
            board_owner=state.board_owner.at[src_hex].set(-1).at[dst_hex].set(o),
            board_count=state.board_count.at[src_hex].set(0).at[dst_hex].set(c),
        )
    
    def _do_attack(self, state: GameState, src_hex: Array, direction: Array, unit_type: Array) -> GameState:
        p = state.current_player
        dst_hex = NEIGHBOR_MAP[src_hex, direction]
        
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=TRUE)
        
        enemy_unit_type = state.board_unit_type[dst_hex]
        new_count = state.board_count[dst_hex] - 1
        is_dead = new_count <= 0
        
        state = state._replace(cemetery=state.cemetery.at[1-p, enemy_unit_type].add(1))
        return state._replace(
            board_count=state.board_count.at[dst_hex].set(jnp.maximum(0, new_count)),
            board_unit_type=jnp.where(is_dead, state.board_unit_type.at[dst_hex].set(-1), state.board_unit_type),
            board_owner=jnp.where(is_dead, state.board_owner.at[dst_hex].set(-1), state.board_owner),
        )
    
    def _do_control(self, state: GameState, hex_idx: Array, unit_type: Array) -> GameState:
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=TRUE)
        return state._replace(control_owner=state.control_owner.at[hex_idx].set(state.current_player))
    
    def _do_recruit(self, state: GameState, target_unit: Array, unit_type: Array) -> GameState:
        p = state.current_player
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=FALSE)
        return state._replace(
            supply=state.supply.at[p, target_unit].add(-1),
            discard_faceup=state.discard_faceup.at[p, target_unit].add(1),
        )
    
    def _do_initiative(self, state: GameState, unit_type: Array) -> GameState:
        state = self._consume_coin(state, unit_type)
        state = self._add_to_discard(state, unit_type, face_up=FALSE)
        return state._replace(initiative_player=state.current_player, initiative_taken_this_round=TRUE)
    
    def _do_pass(self, state: GameState, unit_type: Array) -> GameState:
        state = self._consume_coin(state, unit_type)
        return self._add_to_discard(state, unit_type, face_up=FALSE)
    
    def _check_victory(self, state: GameState) -> GameState:
        p0_win = jnp.sum((state.control_owner == P0) & IS_CONTROL_POINT) >= MAX_CONTROL_MARKERS
        p1_win = jnp.sum((state.control_owner == P1) & IS_CONTROL_POINT) >= MAX_CONTROL_MARKERS
        return state._replace(
            terminated=p0_win | p1_win,
            winner=jnp.where(p0_win, P0, jnp.where(p1_win, P1, jnp.int32(-1)))
        )
    
    def _advance_turn(self, state: GameState, key: Array) -> GameState:
        p = state.current_player
        opp = 1 - p
        both_empty = (jnp.sum(state.hand[p]) == 0) & (jnp.sum(state.hand[opp]) == 0)
        
        return lax.cond(both_empty,
                        lambda: self._draw_phase(state, key),
                        lambda: state._replace(
                            turn_in_round=state.turn_in_round + 1,
                            current_player=jnp.where(jnp.sum(state.hand[opp]) == 0, p, opp)
                        ))

    # =========================================================================
    # VECTORIZED LEGAL ACTION MASK (O(1) logic, zero sequential loops)
    # =========================================================================
    def legal_action_mask(self, state: GameState) -> Array:
        p = state.current_player
        opp = 1 - p
        hand = state.hand[p]
        has_any_coin = jnp.sum(hand) > 0
        colored_hand = hand[:4] > 0 # (4,)

        # Pre-computations
        is_mine = state.board_owner == p # (37,)
        is_empty = state.board_unit_type == -1 # (37,)
        my_unit_types = state.board_unit_type # (37,)
        safe_my_units = jnp.maximum(my_unit_types, 0)
        have_matching_coin = (hand[safe_my_units] > 0) & (my_unit_types >= 0) # (37,)
        can_use_hex = is_mine & have_matching_coin # (37,)

        # 1. Deploy / Bolster (0-147)
        # Deploy: Hex empty, hex controlled by me, unit not on board, unit in hand
        unit_on_board = jax.vmap(lambda u: jnp.any((state.board_owner == p) & (state.board_unit_type == u)))(jnp.arange(4))
        can_deploy_hex = is_empty & (state.control_owner == p) # (37,)
        can_deploy_unit = colored_hand & ~unit_on_board # (4,)
        deploy_mask = can_deploy_hex[:, None] & can_deploy_unit[None, :] # (37, 4)

        # Bolster: Hex is mine, matches my unit type, unit in hand
        valid_types = (my_unit_types >= 0) & (my_unit_types < 4)
        unit_one_hot = jax.nn.one_hot(safe_my_units, 4, dtype=jnp.bool_) & valid_types[:, None]
        bolster_mask = is_mine[:, None] & unit_one_hot & colored_hand[None, :] # (37, 4)

        deploy_bolster_mask = (deploy_mask | bolster_mask).flatten() # (148,)

        # 2. Move (148-369)
        # Source is mine & have matching coin. Dest is valid & empty.
        valid_dst = NEIGHBOR_MAP >= 0 # (37, 6)
        safe_dst = jnp.maximum(NEIGHBOR_MAP, 0)
        dst_empty = state.board_unit_type[safe_dst] == -1 # (37, 6)
        move_mask = can_use_hex[:, None] & valid_dst & dst_empty # (37, 6)
        move_mask = move_mask.flatten() # (222,)

        # 3. Attack (370-591)
        # Source is mine & have matching coin. Dest is valid & has enemy.
        dst_enemy = state.board_owner[safe_dst] == opp # (37, 6)
        attack_mask = can_use_hex[:, None] & valid_dst & dst_enemy # (37, 6)
        attack_mask = attack_mask.flatten() # (222,)

        # 4. Control (592-628)
        # Source is mine & have matching coin. Hex is control point & not already mine.
        can_flip = state.control_owner != p # (37,)
        control_mask = can_use_hex & can_flip & IS_CONTROL_POINT # (37,)

        # 5. Recruit (629-648)
        # Target in supply (colored). Discard coin in hand (any).
        has_supply = state.supply[p, :4] > 0 # (4,)
        has_discard = hand > 0 # (5,)
        recruit_mask = has_supply[:, None] & has_discard[None, :] # (4, 5)
        recruit_mask = recruit_mask.flatten() # (20,)

        # 6. Initiative (649-653)
        can_take_init = (state.initiative_player != p) & ~state.initiative_taken_this_round
        initiative_mask = can_take_init & has_discard # (5,)

        # 7. Pass (654-658)
        pass_mask = has_discard # (5,)

        # Concatenate all vectorized logic
        full_mask = jnp.concatenate([
            deploy_bolster_mask,
            move_mask,
            attack_mask,
            control_mask,
            recruit_mask,
            initiative_mask,
            pass_mask
        ])
        
        return full_mask & has_any_coin
    
    def is_terminal(self, state: GameState) -> Array:
        return state.terminated
    
    def rewards(self, state: GameState) -> Array:
        return lax.cond(
            state.winner >= 0,
            lambda: jnp.float32([-1.0, -1.0]).at[state.winner].set(1.0),
            lambda: jnp.zeros(2, dtype=jnp.float32)
        )
    
    def observe(self, state: GameState, player_id: Array) -> Array:
        opp = 1 - player_id
        
        obs_board = jnp.zeros((NUM_HEXES, NUM_HEX_CHANNELS), dtype=jnp.float32)
        
        for ut in range(NUM_COLORED_UNITS):
            obs_board = obs_board.at[:, ut].set(
                jnp.where((state.board_owner == player_id) & (state.board_unit_type == ut), state.board_count.astype(jnp.float32), 0.0)
            )
            obs_board = obs_board.at[:, NUM_COLORED_UNITS + ut].set(
                jnp.where((state.board_owner == opp) & (state.board_unit_type == ut), state.board_count.astype(jnp.float32), 0.0)
            )
        
        obs_board = obs_board.at[:, 8].set((state.control_owner == player_id).astype(jnp.float32))
        obs_board = obs_board.at[:, 9].set((state.control_owner == opp).astype(jnp.float32))
        obs_board = obs_board.at[:, 10].set(IS_CONTROL_POINT.astype(jnp.float32))
        
        obs_grid = jnp.zeros((GRID_SIZE, GRID_SIZE, GRID_CHANNELS), dtype=jnp.float32)
        obs_grid = obs_grid.at[HEX_TO_GRID_ROW, HEX_TO_GRID_COL, :NUM_HEX_CHANNELS].set(obs_board)
        
        global_features = jnp.concatenate([
            state.hand[player_id].astype(jnp.float32),
            state.discard_faceup[opp].astype(jnp.float32),
            state.supply[player_id].astype(jnp.float32),
            state.supply[opp].astype(jnp.float32),
            state.cemetery[player_id].astype(jnp.float32),
            state.cemetery[opp].astype(jnp.float32),
            state.discard_faceup[player_id].astype(jnp.float32),
            state.discard_facedown[player_id].astype(jnp.float32),
            state.bag[player_id].astype(jnp.float32),
            jnp.array([(state.current_player == player_id).astype(jnp.float32),
                       (state.initiative_player == player_id).astype(jnp.float32),
                       state.initiative_taken_this_round.astype(jnp.float32)]),
            jnp.array([jnp.sum(state.discard_facedown[opp]).astype(jnp.float32),
                       jnp.sum(state.bag[opp]).astype(jnp.float32),
                       jnp.sum(state.hand[opp]).astype(jnp.float32),
                       jnp.sum(state.hand[player_id]).astype(jnp.float32)]),
        ])
        
        global_planes = jnp.broadcast_to(global_features[None, None, :], (GRID_SIZE, GRID_SIZE, NUM_GLOBAL_FEATURES))
        global_planes = global_planes * VALID_HEX_MASK[:, :, None]
        
        obs_grid = obs_grid.at[:, :, NUM_HEX_CHANNELS:].set(global_planes)
        return obs_grid