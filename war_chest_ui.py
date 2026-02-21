import pygame
import math
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp

# Import from the game logic file directly (same directory)
from pgx._src.games.war_chest_simplified import (
    Game, GameState,
    HEX_COORDS, NEIGHBOR_MAP, IS_CONTROL_POINT, NUM_HEXES, NUM_UNIT_TYPES,
    NUM_COLORED_UNITS, NUM_ACTIONS,
    ACT_DEPLOY_START, ACT_DEPLOY_END, ACT_MOVE_START, ACT_MOVE_END,
    ACT_ATTACK_START, ACT_ATTACK_END, ACT_CONTROL_START, ACT_CONTROL_END,
    ACT_RECRUIT_START, ACT_RECRUIT_END,
    ACT_INITIATIVE_START, ACT_INITIATIVE_END,
    ACT_PASS_START, ACT_PASS_END,
    UNIT_RED, UNIT_GREEN, UNIT_BLUE, UNIT_YELLOW, UNIT_ROYAL,
    LOCS_P0_START, LOCS_P1_START, CONTROL_POINT_HEXES,
)

# =============================================================================
# VISUAL CONSTANTS
# =============================================================================

BACKGROUND_COLOR = (230, 220, 190)
HEX_COLOR = (180, 160, 120)
HEX_BORDER = (90, 80, 60)
TEXT_COLOR = (255, 255, 255)
PANEL_BG = (0, 18, 68)
LABEL_COLOR = (200, 200, 200)

HIGHLIGHT_MOVE = (100, 255, 100, 150)
HIGHLIGHT_ATTACK = (255, 100, 100, 150)
HIGHLIGHT_SELECT = (255, 255, 0)

RAVEN_COLOR = (140, 133, 139)
WOLF_COLOR = (215, 178, 82)


UNIT_COLORS = {
    UNIT_RED: (255, 50, 50),
    UNIT_GREEN: (50, 200, 50),
    UNIT_BLUE: (50, 100, 255),
    UNIT_YELLOW: (220, 200, 50),
    UNIT_ROYAL: (200, 180, 50),
}

UNIT_NAMES = ["Red", "Green", "Blue", "Yellow", "Royal"]

# Action types
ACT_TYPE_DEPLOY = 0
ACT_TYPE_MOVE = 1
ACT_TYPE_ATTACK = 2
ACT_TYPE_CONTROL = 3
ACT_TYPE_RECRUIT = 4
ACT_TYPE_INITIATIVE = 5
ACT_TYPE_PASS = 6


def decode_action_py(action: int):
    """Decode action into type and parameters (pure Python version).

    Current action encoding (659 total):
      Deploy:     ACT_DEPLOY_START     + hex_idx * 4 + deploy_unit   (148)
      Move:       ACT_MOVE_START       + hex_idx * 6 + direction     (222)
      Attack:     ACT_ATTACK_START     + hex_idx * 6 + direction     (222)
      Control:    ACT_CONTROL_START    + hex_idx                     (37)
      Recruit:    ACT_RECRUIT_START    + recruit_target * 5 + discard_coin (20)
      Initiative: ACT_INITIATIVE_START + discard_coin                (5)
      Pass:       ACT_PASS_START       + discard_coin                (5)

    Returns:
        (action_type, hex_idx, direction, deploy_unit, recruit_target, discard_coin)
    """
    if ACT_DEPLOY_START <= action < ACT_DEPLOY_END:
        rel = action - ACT_DEPLOY_START
        hex_idx = rel // NUM_COLORED_UNITS
        deploy_unit = rel % NUM_COLORED_UNITS
        return ACT_TYPE_DEPLOY, hex_idx, -1, deploy_unit, -1, -1

    if ACT_MOVE_START <= action < ACT_MOVE_END:
        rel = action - ACT_MOVE_START
        return ACT_TYPE_MOVE, rel // 6, rel % 6, -1, -1, -1

    if ACT_ATTACK_START <= action < ACT_ATTACK_END:
        rel = action - ACT_ATTACK_START
        return ACT_TYPE_ATTACK, rel // 6, rel % 6, -1, -1, -1

    if ACT_CONTROL_START <= action < ACT_CONTROL_END:
        return ACT_TYPE_CONTROL, action - ACT_CONTROL_START, -1, -1, -1, -1

    if ACT_RECRUIT_START <= action < ACT_RECRUIT_END:
        rel = action - ACT_RECRUIT_START
        recruit_target = rel // NUM_UNIT_TYPES
        discard_coin = rel % NUM_UNIT_TYPES
        return ACT_TYPE_RECRUIT, -1, -1, -1, recruit_target, discard_coin

    if ACT_INITIATIVE_START <= action < ACT_INITIATIVE_END:
        discard_coin = action - ACT_INITIATIVE_START
        return ACT_TYPE_INITIATIVE, -1, -1, -1, -1, discard_coin

    if ACT_PASS_START <= action < ACT_PASS_END:
        discard_coin = action - ACT_PASS_START
        return ACT_TYPE_PASS, -1, -1, -1, -1, discard_coin

    return -1, -1, -1, -1, -1, -1


def get_neighbor_hex(hex_idx: int, direction: int) -> int:
    """Get neighbor hex index, returns -1 if off-board."""
    return int(NEIGHBOR_MAP[hex_idx, direction])


class WarChestUI:
    def __init__(self, human_player: int = 1):
        """
        Initialize the UI.

        Args:
            human_player: 0 for Wolf (P0), 1 for Raven (P1)
        """
        pygame.init()
        self.width, self.height = 1920, 1080
        self.hex_size = 55
        self.panel_width = 450

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("War Chest Simplified (JAX)")

        # Fonts
        self.font = pygame.font.SysFont("Arial", 20, bold=True)
        self.header_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.small_font = pygame.font.SysFont("Arial", 16)
        self.count_font = pygame.font.SysFont("Arial", 28, bold=True)

        # JAX Game Engine — JIT-compile hot paths for snappy UI
        self.game = Game()
        self._jit_init = jax.jit(self.game.init)
        self._jit_step = jax.jit(self.game.step)
        self._jit_legal = jax.jit(self.game.legal_action_mask)
        self.key = jax.random.PRNGKey(42)
        self.state = None
        self.reset_game()

        # Human plays as...
        self.human_player = human_player  # 0=Wolf, 1=Raven

        # Calculate board offset for centering
        self._calculate_offset()

        # UI State
        self.selected_unit_type = None  # Which unit type is selected from hand
        self.legal_actions = []  # List of legal action indices
        self.hand_rects = []  # (unit_type, rect) for hand coins
        self.supply_rects = []  # (action_idx, rect) for supply coins
        self.button_rects = []  # (action_idx, rect) for buttons
        self.action_log = []

        # Load assets
        self.images = {}
        self._load_assets()

    def _calculate_offset(self):
        """Calculate offset to center the board."""
        avg_q = sum(int(HEX_COORDS[i, 0]) for i in range(NUM_HEXES)) / NUM_HEXES
        avg_r = sum(int(HEX_COORDS[i, 1]) for i in range(NUM_HEXES)) / NUM_HEXES
        cx, cy = self._axial_to_pixel_calc(avg_q, avg_r)
        self.offset = (self.width // 2 - cx, self.height // 2 - cy)

    def _axial_to_pixel_calc(self, q, r):
        """Convert axial coordinates to pixel position (before offset)."""
        x = self.hex_size * 1.5 * q
        y = -1 * (self.hex_size * math.sqrt(3) * (r + q / 2.0))
        return x, y

    def axial_to_pixel(self, q, r):
        """Convert axial coordinates to screen pixel position."""
        x, y = self._axial_to_pixel_calc(q, r)
        return int(x + self.offset[0]), int(y + self.offset[1])

    def hex_idx_to_pixel(self, hex_idx):
        """Convert hex index to screen pixel position."""
        q, r = int(HEX_COORDS[hex_idx, 0]), int(HEX_COORDS[hex_idx, 1])
        return self.axial_to_pixel(q, r)

    def pixel_to_hex_idx(self, px, py):
        """Convert pixel to nearest hex index, returns -1 if not on board."""
        x = px - self.offset[0]
        y = py - self.offset[1]

        # Convert to axial
        q = (2.0 / 3 * x) / self.hex_size
        r = (-1.0 / 3 * x - math.sqrt(3) / 3 * y) / self.hex_size

        # Round to nearest hex
        q, r = self._hex_round(q, r)

        # Find matching hex index
        for i in range(NUM_HEXES):
            if int(HEX_COORDS[i, 0]) == q and int(HEX_COORDS[i, 1]) == r:
                return i
        return -1

    def _hex_round(self, q, r):
        """Round fractional axial coordinates to nearest hex."""
        s = -q - r
        rq, rr, rs = round(q), round(r), round(s)
        q_diff = abs(rq - q)
        r_diff = abs(rr - r)
        s_diff = abs(rs - s)
        if q_diff > r_diff and q_diff > s_diff:
            rq = -rr - rs
        elif r_diff > s_diff:
            rr = -rq - rs
        return int(rq), int(rr)

    def _load_assets(self):
        """Load image assets."""
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")

        def load_img(subpath, filename, size_factor=1.6):
            path = os.path.join(base_path, subpath, filename)
            if os.path.exists(path):
                img = pygame.image.load(path).convert_alpha()
                target_size = int(self.hex_size * size_factor)
                return pygame.transform.smoothscale(img, (target_size, target_size))
            return None

        # Unit coins
        self.images[UNIT_RED] = load_img("coin", "Red.png")
        self.images[UNIT_GREEN] = load_img("coin", "Green.png")
        self.images[UNIT_BLUE] = load_img("coin", "Blue.png")
        self.images[UNIT_YELLOW] = load_img("coin", "Yellow.png")
        self.images["RavenRoyal"] = load_img("coin", "RavenRoyalCoin.png")
        self.images["WolfRoyal"] = load_img("coin", "WolfRoyalCoin.png")
        self.images["Back"] = load_img("coin", "CoinReverse.png")

        # Control markers
        self.images["ControlNeutral"] = load_img("control", "NeutralControlMarker.png", 1.8)
        self.images["ControlRaven"] = load_img("control", "RavenControlMarker.png", 1.8)
        self.images["ControlWolf"] = load_img("control", "WolfControlMarker.png", 1.8)

    def reset_game(self):
        """Reset to a new game."""
        self.key, subkey = jax.random.split(self.key)
        self.state = self._jit_init(subkey)
        self.selected_unit_type = None
        self.legal_actions = []
        self.action_log = []

    @property
    def x(self):
        """Shorthand for game state."""
        return self.state

    @property
    def terminated(self):
        """Check if game is terminated."""
        return bool(self.state.terminated)

    def get_legal_action_indices(self):
        """Get list of legal action indices."""
        mask = np.array(self._jit_legal(self.state))
        return [i for i in range(NUM_ACTIONS) if mask[i]]

    def get_hand_as_list(self, player):
        """Convert histogram hand to list of unit types."""
        hand_hist = np.array(self.state.hand[player])
        result = []
        for ut in range(NUM_UNIT_TYPES):
            result.extend([ut] * int(hand_hist[ut]))
        return result

    def step(self, action: int):
        """Execute an action."""
        self.key, subkey = jax.random.split(self.key)
        self.state = self._jit_step(self.state, jnp.int32(action), subkey)
        self.selected_unit_type = None
        self.legal_actions = []

    # =========================================================================
    # DRAWING
    # =========================================================================

    def draw_hex(self, hex_idx, color, width=0):
        """Draw a hexagon at the given hex index."""
        x, y = self.hex_idx_to_pixel(hex_idx)
        points = []
        for i in range(6):
            angle_rad = math.pi / 180 * (60 * i)
            px = x + self.hex_size * math.cos(angle_rad)
            py = y + self.hex_size * math.sin(angle_rad)
            points.append((px, py))

        if len(color) == 4:  # RGBA
            surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(surface, color, points, width)
            self.screen.blit(surface, (0, 0))
        else:
            pygame.draw.polygon(self.screen, color, points, width)

    def draw_coin_at(self, x, y, unit_type, player=None):
        """Draw a coin at the given position."""
        if unit_type == UNIT_ROYAL:
            img_key = "RavenRoyal" if player == 1 else "WolfRoyal"
            img = self.images.get(img_key)
        else:
            img = self.images.get(unit_type)

        if img:
            rect = img.get_rect(center=(x, y))
            self.screen.blit(img, rect)
            return rect
        else:
            # Fallback circle
            col = UNIT_COLORS.get(unit_type, (100, 100, 100))
            pygame.draw.circle(self.screen, col, (x, y), 30)
            name = UNIT_NAMES[unit_type][:2] if unit_type < len(UNIT_NAMES) else "?"
            txt = self.font.render(name, True, (255, 255, 255))
            self.screen.blit(txt, (x - 10, y - 10))
            return pygame.Rect(x - 30, y - 30, 60, 60)

    def draw_count_badge(self, x, y, count):
        """Draw a count badge near a coin."""
        if count < 0:
            return
        txt = self.count_font.render(str(count), True, (255, 255, 0))
        shadow = self.count_font.render(str(count), True, (0, 0, 0))
        tx, ty = x + 20, y + 15
        self.screen.blit(shadow, (tx + 2, ty + 2))
        self.screen.blit(txt, (tx, ty))

    def render(self):
        """Render the full game state."""
        self.screen.fill(BACKGROUND_COLOR)
        self.hand_rects = []
        self.supply_rects = []
        self.button_rects = []

        # Draw board
        self._draw_board()

        # Draw highlights
        if self.selected_unit_type is not None and int(self.x.current_player) == self.human_player:
            self._draw_highlights()

        # Draw panels
        self._draw_player_panel(1, 0, 0, self.panel_width, self.height)  # Raven on left
        self._draw_player_panel(0, self.width - self.panel_width, 0, self.panel_width, self.height)  # Wolf on right

        # Draw hand HUD
        self._draw_hand_hud()

        # Draw action buttons
        if int(self.x.current_player) == self.human_player and not self.terminated:
            self._draw_action_buttons()

        # Draw game over
        if self.terminated:
            self._draw_game_over()

        # Draw action log
        self._draw_action_log()

        pygame.display.flip()

    def _draw_board(self):
        """Draw the hex grid and units."""
        # Draw all hexes
        for i in range(NUM_HEXES):
            self.draw_hex(i, HEX_COLOR, 0)
            self.draw_hex(i, HEX_BORDER, 3)

        # Draw control markers
        for i in range(NUM_HEXES):
            if not bool(IS_CONTROL_POINT[i]):
                continue
            owner = int(self.x.control_owner[i])
            x, y = self.hex_idx_to_pixel(i)
            if owner == 0:
                img = self.images.get("ControlWolf")
            elif owner == 1:
                img = self.images.get("ControlRaven")
            else:
                img = self.images.get("ControlNeutral")
            if img:
                self.screen.blit(img, img.get_rect(center=(x, y)))

        # Draw units
        for i in range(NUM_HEXES):
            unit_type = int(self.x.board_unit_type[i])
            if unit_type >= 0:
                owner = int(self.x.board_owner[i])
                count = int(self.x.board_count[i])
                x, y = self.hex_idx_to_pixel(i)
                self.draw_coin_at(x, y, unit_type, owner)
                self.draw_count_badge(x, y, count)

    def _draw_highlights(self):
        """Draw action highlights based on selected coin."""
        if not self.legal_actions:
            self._update_legal_actions()

        for action_idx in self.legal_actions:
            act_type, hex_idx, direction, deploy_unit, _, _ = decode_action_py(action_idx)

            if act_type == ACT_TYPE_DEPLOY:
                if int(self.x.board_unit_type[hex_idx]) >= 0:
                    # Bolster - draw ring
                    x, y = self.hex_idx_to_pixel(hex_idx)
                    pygame.draw.circle(self.screen, HIGHLIGHT_SELECT, (x, y), 35, 3)
                else:
                    # Deploy - green highlight
                    self.draw_hex(hex_idx, HIGHLIGHT_MOVE, 0)

            elif act_type == ACT_TYPE_MOVE:
                dst = get_neighbor_hex(hex_idx, direction)
                if dst >= 0:
                    self.draw_hex(dst, HIGHLIGHT_MOVE, 0)

            elif act_type == ACT_TYPE_ATTACK:
                dst = get_neighbor_hex(hex_idx, direction)
                if dst >= 0:
                    self.draw_hex(dst, HIGHLIGHT_ATTACK, 0)

    def _update_legal_actions(self):
        """Update legal actions based on selected unit type."""
        if self.selected_unit_type is None:
            self.legal_actions = []
            return

        all_legal = self.get_legal_action_indices()

        # Filter by selected unit type
        filtered = []
        for action_idx in all_legal:
            if self._action_uses_unit_type(action_idx, self.selected_unit_type):
                filtered.append(action_idx)

        self.legal_actions = filtered

    def _action_uses_unit_type(self, action_idx: int, unit_type: int) -> bool:
        """Check if an action can be performed with the given unit type."""
        act_type, hex_idx, direction, deploy_unit, recruit_target, discard_coin = decode_action_py(action_idx)
        player = int(self.x.current_player)

        if act_type == ACT_TYPE_DEPLOY:
            # Deploy/bolster: the deploy_unit encoded in the action must match
            return deploy_unit == unit_type

        if act_type in (ACT_TYPE_MOVE, ACT_TYPE_ATTACK, ACT_TYPE_CONTROL):
            # Must have matching unit at source hex
            board_ut = int(self.x.board_unit_type[hex_idx])
            board_owner = int(self.x.board_owner[hex_idx])
            return board_ut == unit_type and board_owner == player

        if act_type == ACT_TYPE_RECRUIT:
            # The discard_coin encoded in the action must match the selected unit
            return discard_coin == unit_type

        if act_type == ACT_TYPE_INITIATIVE:
            # The discard_coin encoded in the action must match
            return discard_coin == unit_type

        if act_type == ACT_TYPE_PASS:
            # The discard_coin encoded in the action must match
            return discard_coin == unit_type

        return False

    def _draw_player_panel(self, player, x, y, w, h):
        """Draw a player's resource panel."""
        pygame.draw.rect(self.screen, PANEL_BG, (x, y, w, h))
        pygame.draw.rect(self.screen, (100, 100, 100), (x, y, w, h), 2)

        color = WOLF_COLOR if player == 0 else RAVEN_COLOR
        name = "Wolf" if player == 0 else "Raven"

        PANEL_LEFT_PAD = 20
        COIN_ROW_HEIGHT = 80
        COIN_STEP_X = 95
        SECTION_SPACING = 30

        current_y = y + 20

        # Title
        title = self.header_font.render(name, True, color)
        self.screen.blit(title, (x + PANEL_LEFT_PAD, current_y))
        current_y += 40

        # Supply
        self._draw_section(player, x, current_y, "Supply", self.x.supply[player], PANEL_LEFT_PAD, COIN_STEP_X)
        current_y += COIN_ROW_HEIGHT + SECTION_SPACING + 50

        # Discard
        total_down = int(np.sum(np.array(self.x.discard_facedown[player])))
        self._draw_discard_section(player, x, current_y, self.x.discard_faceup[player], total_down, PANEL_LEFT_PAD, COIN_STEP_X)
        current_y += COIN_ROW_HEIGHT + SECTION_SPACING + 50

        # Cemetery
        self._draw_section(player, x, current_y, "Cemetery", self.x.cemetery[player], PANEL_LEFT_PAD, COIN_STEP_X)
        current_y += COIN_ROW_HEIGHT + SECTION_SPACING + 50

        # Bag
        lbl = self.font.render("Bag", True, TEXT_COLOR)
        self.screen.blit(lbl, (x + PANEL_LEFT_PAD, current_y))
        current_y += 50
        total_bag = int(np.sum(np.array(self.x.bag[player])))
        draw_x = x + PANEL_LEFT_PAD + 40
        row_y = current_y + 40
        back_img = self.images.get("Back")
        if back_img:
            self.screen.blit(back_img, back_img.get_rect(center=(draw_x, row_y)))
        else:
            pygame.draw.circle(self.screen, (80, 80, 80), (draw_x, row_y), 30)
        self.draw_count_badge(draw_x, row_y, total_bag)

    def _draw_section(self, player, panel_x, y, title, histogram, pad, step):
        """Draw a section with coin counts."""
        lbl = self.font.render(title, True, TEXT_COLOR)
        self.screen.blit(lbl, (panel_x + pad, y))

        draw_x = panel_x + pad + 40
        row_y = y + 50 + 40

        hist = np.array(histogram)
        for ut in range(NUM_COLORED_UNITS):  # Only colored units in supply/cemetery
            count = int(hist[ut])
            if count > 0 or title == "Supply":
                rect = self.draw_coin_at(draw_x, row_y, ut, player)
                self.draw_count_badge(draw_x, row_y, count)

                # Check if this is recruitable (only for human's supply panel)
                if title == "Supply" and player == self.human_player:
                    if self.selected_unit_type is not None and count > 0:
                        # Find any recruit action targeting this unit with our selected discard coin
                        recruit_action = ACT_RECRUIT_START + ut * NUM_UNIT_TYPES + self.selected_unit_type
                        if recruit_action in self.legal_actions:
                            pygame.draw.circle(self.screen, HIGHLIGHT_SELECT, rect.center, 32, 3)
                            self.supply_rects.append((recruit_action, rect))

                draw_x += step

    def _draw_discard_section(self, player, panel_x, y, faceup_hist, facedown_count, pad, step):
        """Draw discard pile section."""
        lbl = self.font.render("Discard", True, TEXT_COLOR)
        self.screen.blit(lbl, (panel_x + pad, y))

        draw_x = panel_x + pad + 40
        row_y = y + 50 + 40

        # Face-down pile
        if facedown_count > 0:
            back_img = self.images.get("Back")
            if back_img:
                self.screen.blit(back_img, back_img.get_rect(center=(draw_x, row_y)))
            else:
                pygame.draw.circle(self.screen, (80, 80, 80), (draw_x, row_y), 30)
            self.draw_count_badge(draw_x, row_y, facedown_count)
            draw_x += step

        # Face-up coins
        hist = np.array(faceup_hist)
        for ut in range(NUM_UNIT_TYPES):
            count = int(hist[ut])
            if count > 0:
                self.draw_coin_at(draw_x, row_y, ut, player)
                self.draw_count_badge(draw_x, row_y, count)
                draw_x += step

    def _draw_hand_hud(self):
        """Draw both players' hands."""
        avail_w = self.width - 2 * self.panel_width
        cx = self.panel_width + avail_w // 2

        # Opponent hand (face down) at top
        opp = 1 - self.human_player
        opp_hand = self.get_hand_as_list(opp)
        opp_y = 120

        lbl = self.header_font.render("Opponent Hand", True, TEXT_COLOR)
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, opp_y - 50))

        opp_start_x = cx - (len(opp_hand) * 80) // 2
        back_img = self.images.get("Back")
        for i, ut in enumerate(opp_hand):
            x_pos = opp_start_x + i * 90
            if back_img:
                self.screen.blit(back_img, back_img.get_rect(center=(x_pos + 40, opp_y + 40)))

        # Human hand (face up) at bottom
        human_hand = self.get_hand_as_list(self.human_player)
        hand_y = self.height - 100

        txt = "Your Hand"
        if int(self.x.current_player) != self.human_player:
            txt = "Your Hand (Waiting...)"
        lbl = self.header_font.render(txt, True, TEXT_COLOR)
        self.screen.blit(lbl, (cx - lbl.get_width() // 2, hand_y - 50))

        start_x = cx - (len(human_hand) * 80) // 2
        for i, ut in enumerate(human_hand):
            x_pos = start_x + i * 90
            rect = self.draw_coin_at(x_pos + 40, hand_y + 40, ut, self.human_player)

            if int(self.x.current_player) == self.human_player:
                self.hand_rects.append((ut, rect))
                if ut == self.selected_unit_type:
                    pygame.draw.circle(self.screen, HIGHLIGHT_SELECT, rect.center, 35, 3)

        # Initiative indicator
        init_player = int(self.x.initiative_player)
        init_name = "Wolf" if init_player == 0 else "Raven"
        init_txt = f"Initiative: {init_name}"
        init_lbl = self.header_font.render(init_txt, True, TEXT_COLOR)
        self.screen.blit(init_lbl, (self.panel_width + 20, 20))

        # Current player indicator
        curr = int(self.x.current_player)
        curr_name = "Wolf" if curr == 0 else "Raven"
        curr_txt = f"Current Turn: {curr_name}"
        curr_lbl = self.font.render(curr_txt, True, TEXT_COLOR)
        self.screen.blit(curr_lbl, (self.panel_width + 20, 55))

    def _draw_action_buttons(self):
        """Draw action buttons for Pass, Initiative, Control."""
        if self.selected_unit_type is None:
            return

        buttons = []
        seen_types = set()

        for action_idx in self.legal_actions:
            act_type, hex_idx, _, _, _, discard_coin = decode_action_py(action_idx)

            if act_type == ACT_TYPE_PASS and 'Pass' not in seen_types:
                buttons.append(('Pass', action_idx))
                seen_types.add('Pass')

            elif act_type == ACT_TYPE_INITIATIVE and 'Initiative' not in seen_types:
                buttons.append(('Claim Initiative', action_idx))
                seen_types.add('Initiative')

            elif act_type == ACT_TYPE_CONTROL:
                q, r = int(HEX_COORDS[hex_idx, 0]), int(HEX_COORDS[hex_idx, 1])
                key = f'Control_{hex_idx}'
                if key not in seen_types:
                    buttons.append((f'Control ({q},{r})', action_idx))
                    seen_types.add(key)

        start_x = self.width // 2 - 200
        y = self.height - 160

        for label, action_idx in buttons:
            rect = pygame.Rect(start_x, y, 180, 50)
            pygame.draw.rect(self.screen, (70, 70, 70), rect)
            pygame.draw.rect(self.screen, (200, 200, 200), rect, 2)

            txt = self.font.render(label, True, TEXT_COLOR)
            self.screen.blit(txt, (rect.centerx - txt.get_width() // 2,
                                  rect.centery - txt.get_height() // 2))

            self.button_rects.append((action_idx, rect))
            start_x += 200

    def _draw_game_over(self):
        """Draw game over overlay."""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 200))
        self.screen.blit(surface, (0, 0))

        winner = int(self.x.winner)
        winner_name = "Wolf" if winner == 0 else "Raven"
        txt = f"GAME OVER! Winner: {winner_name}"
        lbl = self.header_font.render(txt, True, (255, 215, 0))
        scaled = pygame.transform.scale(lbl, (lbl.get_width() * 2, lbl.get_height() * 2))
        self.screen.blit(scaled, (self.width // 2 - scaled.get_width() // 2,
                                  self.height // 2 - scaled.get_height() // 2))

    def _draw_action_log(self):
        """Draw recent action log."""
        x = 1450 - 320
        y = 20

        surface = pygame.Surface((300, 180), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 150))
        self.screen.blit(surface, (x, y))

        curr_y = y + 10
        for line in self.action_log[-8:]:
            col = (200, 200, 200)
            if "[Wolf]" in line:
                col = WOLF_COLOR
            elif "[Raven]" in line:
                col = RAVEN_COLOR

            if len(line) > 43:
                line = line[:40] + "..."
            txt = self.small_font.render(line, True, col)
            self.screen.blit(txt, (x + 10, curr_y))
            curr_y += 20

    # =========================================================================
    # INPUT HANDLING
    # =========================================================================

    def handle_input(self):
        """Handle user input. Returns False to quit."""
        current_player = int(self.x.current_player)

        # If game over, just handle quit
        if self.terminated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_r:
                        self.reset_game()
            return True

        # AI turn
        if current_player != self.human_player:
            pygame.time.delay(500)
            pygame.event.pump()
            self._do_ai_turn()
            return True

        # Human turn
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_r:
                    self.reset_game()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = pygame.mouse.get_pos()

                # Check hand clicks
                for ut, rect in self.hand_rects:
                    if rect.collidepoint(mx, my):
                        self.selected_unit_type = ut
                        self._update_legal_actions()
                        return True

                if self.selected_unit_type is not None:
                    # Check board clicks
                    hex_idx = self.pixel_to_hex_idx(mx, my)
                    if hex_idx >= 0:
                        action = self._find_action_for_hex(hex_idx)
                        if action is not None:
                            self._execute_action(action)
                            return True

                    # Check supply clicks
                    for action_idx, rect in self.supply_rects:
                        if rect.collidepoint(mx, my):
                            self._execute_action(action_idx)
                            return True

                    # Check button clicks
                    for action_idx, rect in self.button_rects:
                        if rect.collidepoint(mx, my):
                            self._execute_action(action_idx)
                            return True

        return True

    def _find_action_for_hex(self, target_hex):
        """Find a legal action for clicking on a target hex."""
        for action_idx in self.legal_actions:
            act_type, hex_idx, direction, deploy_unit, _, _ = decode_action_py(action_idx)

            if act_type == ACT_TYPE_DEPLOY:
                if hex_idx == target_hex:
                    return action_idx

            elif act_type == ACT_TYPE_MOVE:
                dst = get_neighbor_hex(hex_idx, direction)
                if dst == target_hex:
                    return action_idx

            elif act_type == ACT_TYPE_ATTACK:
                dst = get_neighbor_hex(hex_idx, direction)
                if dst == target_hex:
                    return action_idx

        return None

    def _execute_action(self, action_idx):
        """Execute an action and log it."""
        player_name = "Wolf" if self.human_player == 0 else "Raven"
        action_str = self._action_to_string(action_idx, show_discard=True)
        msg = f"[{player_name}] {action_str}"
        print(msg)
        self.action_log.append(msg)

        self.step(action_idx)

    def _do_ai_turn(self):
        """Execute AI turn (random for now)."""
        legal = self.get_legal_action_indices()
        if legal:
            action_idx = np.random.choice(legal)
            ai_name = "Wolf" if (1 - self.human_player) == 0 else "Raven"
            action_str = self._action_to_string(action_idx, show_discard=False)
            msg = f"[{ai_name}] {action_str}"
            print(msg)
            self.action_log.append(msg)
            self.step(action_idx)

    def _action_to_string(self, action_idx, show_discard=True):
        """Convert action index to human-readable string.

        Args:
            action_idx: The action index to convert.
            show_discard: If True, show which coin was discarded face-down.
                          Set to False for opponent actions (hidden info).
        """
        act_type, hex_idx, direction, deploy_unit, recruit_target, discard_coin = decode_action_py(action_idx)

        if act_type == ACT_TYPE_DEPLOY:
            q, r = int(HEX_COORDS[hex_idx, 0]), int(HEX_COORDS[hex_idx, 1])
            unit_name = UNIT_NAMES[deploy_unit]
            board_ut = int(self.x.board_unit_type[hex_idx])
            if board_ut >= 0:
                return f"Bolster {unit_name} at ({q},{r})"
            return f"Deploy {unit_name} at ({q},{r})"

        if act_type == ACT_TYPE_MOVE:
            q, r = int(HEX_COORDS[hex_idx, 0]), int(HEX_COORDS[hex_idx, 1])
            dirs = ["E", "NE", "NW", "W", "SW", "SE"]
            return f"Move from ({q},{r}) {dirs[direction]}"

        if act_type == ACT_TYPE_ATTACK:
            q, r = int(HEX_COORDS[hex_idx, 0]), int(HEX_COORDS[hex_idx, 1])
            dirs = ["E", "NE", "NW", "W", "SW", "SE"]
            return f"Attack from ({q},{r}) {dirs[direction]}"

        if act_type == ACT_TYPE_CONTROL:
            q, r = int(HEX_COORDS[hex_idx, 0]), int(HEX_COORDS[hex_idx, 1])
            return f"Control ({q},{r})"

        if act_type == ACT_TYPE_RECRUIT:
            suffix = f" (discard {UNIT_NAMES[discard_coin]})" if show_discard else ""
            return f"Recruit {UNIT_NAMES[recruit_target]}{suffix}"

        if act_type == ACT_TYPE_INITIATIVE:
            suffix = f" (discard {UNIT_NAMES[discard_coin]})" if show_discard else ""
            return f"Claim Initiative{suffix}"

        if act_type == ACT_TYPE_PASS:
            suffix = f" (discard {UNIT_NAMES[discard_coin]})" if show_discard else ""
            return f"Pass{suffix}"

        return f"Unknown({action_idx})"

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True

        while running:
            running = self.handle_input()
            self.render()
            clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    # Human plays as Raven (player 1)
    ui = WarChestUI(human_player=1)
    ui.run()