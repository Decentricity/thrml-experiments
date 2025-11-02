#!/usr/bin/env python3

robots.py -- multi-robot warehouse routing (pygame visual)

deps: pip install pygame numpy

run:  python robots.py

import random, time from dataclasses import dataclass from typing import List, Tuple, Dict, Optional, Deque from collections import deque

import numpy as np import pygame

--------------------------- config ---------------------------

CELL = 28            # px per grid cell MARGIN = 2           # px between cells FPS = 30 SEED = 42

fleet sizing

ROBOTS_PER_PICK = 3     # spawn this many robots near every green pick station MAX_ROBOTS = 64         # global cap (safety)

energy weights (editable at runtime)

W_STEP = 1.0         # base cost per move W_GOAL = 0.6         # heuristic to goal DEFAULT_W_CONG = 2.0 # congestion weight DEFAULT_BETA = 1.2   # inverse temperature

motion stabilizers

W_TURN = 0.8         # penalty for reversing direction (momentum) W_NEARWALL = 0.3     # mild penalty for stepping next to shelves (keeps lanes centered)

dwell times (frames)

LOAD_DWELL = 45      # time spent loading at pick station (green) UNLOAD_DWELL = 25    # time spent unloading beside shelf

UTIL_DECAY = 0.92    # decay for utilization heatmap

colors

BG = (18, 18, 22) GRID = (40, 40, 48) SHELF = (90, 90, 100) SHELF_FULL = (120, 120, 40) GOAL = (0, 180, 110) TXT = (220, 220, 230) HEAT = (255, 120, 50) CARGO = (240, 220, 120)

ROBOT_COLORS = [ (255, 92, 92), (72, 149, 239), (255, 184, 77), (127, 219, 109), (169, 132, 250), (255, 109, 186), (88, 190, 255), (255, 140, 105) ]

--------------------------- logging ---------------------------

LOG_MAX = 200 LOG: Deque[tuple[str, tuple[int,int,int]]] = deque(maxlen=LOG_MAX)

def log(msg: str, color: tuple[int,int,int] = TXT): ts = time.strftime("%H:%M:%S") LOG.append((f"[{ts}] {msg}", color))

def logc(rb, msg: str): log(msg, getattr(rb, "color", TXT))

--------------------------- maps ---------------------------

def preset_map(kind: int) -> Tuple[np.ndarray, List[Tuple[int,int]], List[Tuple[int,int]]]: """return (obstacles grid), nominal_starts, pick_stations (green). obstacles: 1 == blocked (shelf), 0 == free. kind 3 is our default (aka map 4 in ui). """ if kind == 0: # default small (map 4 layout) w, h = 16, 12 obs = np.zeros((h, w), dtype=np.int8) for r in range(2, h-2, 3): obs[r, :] = 1 obs[:, 7] = 1 obs[3, :] = 0 obs[8, :] = 0 starts = [(1,1),(h-2,w-2),(1,w-2),(h-2,1)] picks  = [(8,1),(8,w-2),(3,1),(3,w-2)] return obs, starts, picks if kind == 1: w, h = 20, 14 obs = np.zeros((h, w), dtype=np.int8) for c in range(2, w-2, 3): obs[:, c] = 1 obs[:, c+1] = 1 obs[3, :] = 0 obs[8, :] = 0 starts = [(0,0),(0,w-1),(h-1,0),(h-1,w-1)] picks  = [(8,1),(8,w-2),(3,1),(3,w-2)] return obs, starts, picks if kind == 2: w, h = 22, 16 obs = np.zeros((h, w), dtype=np.int8) for r in range(2, h-2, 3): obs[r, :] = 1 obs[r+1, :] = 1 obs[:, 5] = 1; obs[:, 6] = 1 obs[:, 15] = 1; obs[:, 16] = 1 obs[1: h-1, 10] = 0 starts = [(1,1),(1,w-2),(h-2,1),(h-2,w-2),(1,10),(h-2,10)] picks  = [(h-2, w-2),(h-2,1),(1,w-2),(1,1),(h-2,10),(1,10)] return obs, starts, picks # kind == 3 w, h = 26, 16 obs = np.zeros((h, w), dtype=np.int8) for c in range(3, w-3): if c % 4 in (0,1,2): obs[:, c] = 1 obs[4,:] = 0; obs[11,:] = 0 starts = [(0,2),(0,w-3),(h-1,2),(h-1,w-3),(0,w//2),(h-1,w//2)] picks  = [(11,1),(11,w-2),(4,1),(4,w-2),(11,w//2),(4,w//2)] return obs, starts, picks

--------------------------- sim types ---------------------------

@dataclass class Robot: id: int pos: Tuple[int,int] goal: Tuple[int,int] color: Tuple[int,int,int] mode: str = "to_pick"          # to_pick, loading, to_drop, unloading dwell: int = 0 last_move: Tuple[int,int] = (0,0) carrying: bool = False drop_shelf: Optional[Tuple[int,int]] = None

def at_goal(self) -> bool:
    return self.pos == self.goal

Move = Tuple[int,int]  # dr, dc MOVES: List[Move] = [(0,0), (-1,0), (1,0), (0,-1), (0,1)]  # stay, N, S, W, E

@dataclass class SimState: obs: np.ndarray shelves_full: np.ndarray robots: List[Robot] picks: List[Tuple[int,int]] util_h: np.ndarray util_v: np.ndarray beta: float w_cong: float reserved: Dict[Tuple[int,int], int]

--------------------------- helpers ---------------------------

def in_bounds(obs: np.ndarray, r: int, c: int) -> bool: h, w = obs.shape return 0 <= r < h and 0 <= c < w

def valid_cell(obs: np.ndarray, r: int, c: int) -> bool: return in_bounds(obs, r, c) and obs[r, c] == 0

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int: return abs(a[0]-b[0]) + abs(a[1]-b[1])

def edge_indices(a: Tuple[int,int], b: Tuple[int,int]): r1, c1 = a; r2, c2 = b if r1 == r2 and c1 == c2: return None, None if r1 == r2: hrow = r1 hcol = min(c1, c2) return None, (hrow, hcol) if c1 == c2: vrow = min(r1, r2) vcol = c1 return (vrow, vcol), None return None, None

def nearwall_penalty(obs: np.ndarray, r: int, c: int) -> float: cnt = 0 for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: rr, cc = r+dr, c+dc if in_bounds(obs, rr, cc) and obs[rr, cc] == 1: cnt += 1 return W_NEARWALL * cnt

def nearest_pick(picks: List[Tuple[int,int]], pos: Tuple[int,int]) -> Tuple[int,int]: return min(picks, key=lambda g: manhattan(g, pos))

def find_drop_target(state: SimState, pos: Tuple[int,int]) -> Tuple[Tuple[int,int], Tuple[int,int]]: candidates: List[Tuple[int,int,int,int,int]] = []  # (dist, sr, sc, fr, fc) h, w = state.obs.shape for r in range(h): for c in range(w): if state.obs[r,c] == 1 and not state.shelves_full[r,c] and (state.reserved.get((r,c)) is None): for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: fr, fc = r+dr, c+dc if valid_cell(state.obs, fr, fc): d = manhattan(pos, (fr,fc)) candidates.append((d, r, c, fr, fc)) break if not candidates: for r in range(h): for c in range(w): if state.obs[r,c] == 1 and (state.reserved.get((r,c)) is None): for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: fr, fc = r+dr, c+dc if valid_cell(state.obs, fr, fc): d = manhattan(pos, (fr,fc)) candidates.append((d, r, c, fr, fc)) break if not candidates: return pos, pos candidates.sort(key=lambda t: t[0]) _, sr, sc, fr, fc = candidates[0] return (sr, sc), (fr, fc)

from collections import deque as _deque

def bfs_free_cells(obs: np.ndarray, start: Tuple[int,int], limit: int) -> List[Tuple[int,int]]: h, w = obs.shape q: Deque[Tuple[int,int]] = _deque([start]) seen = {start} out: List[Tuple[int,int]] = [] while q and len(out) < limit: r, c = q.popleft() if valid_cell(obs, r, c): out.append((r, c)) for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: nr, nc = r+dr, c+dc if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen: seen.add((nr, nc)) q.append((nr, nc)) return out

--------------------------- energy model ---------------------------

def propose_moves(state: SimState) -> Dict[int, Tuple[Tuple[int,int], Tuple[int,int]]]: obs, robots = state.obs, state.robots block_a, block_b = [], [] for rb in robots: (block_a if (rb.pos[0]+rb.pos[1])%2==0 else block_b).append(rb)

chosen: Dict[int, Tuple[Tuple[int,int], Tuple[int,int]]] = {}

def sample_block(block: List[Robot]):
    occupied_next = {rid: pos for rid, (_, pos) in chosen.items()}
    occupied_now = {rb.pos for rb in state.robots}
    for rb in block:
        r, c = rb.pos
        if rb.mode in ("loading","unloading") and rb.dwell > 0:
            chosen[rb.id] = ((0,0), (r,c))
            continue

        cand_positions: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []
        energies: List[float] = []
        for dr, dc in MOVES:
            nr, nc = r+dr, c+dc
            if not valid_cell(obs, nr, nc):
                continue
            # forbid stepping into a cell already occupied (end-of-tick uniqueness)
            if (nr, nc) in occupied_next.values() or ((nr, nc) in occupied_now and not (nr==r and nc==c)):
                penalty = 1e5
            else:
                penalty = 0.0

            e = penalty
            e += W_STEP * (0.0 if (dr==0 and dc==0) else 1.0)
            e += W_GOAL * manhattan((nr,nc), rb.goal)
            if (dr, dc) == (-rb.last_move[0], -rb.last_move[1]) and (dr, dc) != (0,0):
                e += W_TURN
            e += nearwall_penalty(obs, nr, nc)
            v_idx, h_idx = edge_indices((r,c),(nr,nc))
            if v_idx is not None:
                vr, vc = v_idx
                e += state.w_cong * state.util_v[vr, vc]
            if h_idx is not None:
                hr, hc = h_idx
                e += state.w_cong * state.util_h[hr, hc]
            cand_positions.append(((dr,dc),(nr,nc)))
            energies.append(e)
        if not cand_positions:
            cand_positions.append(((0,0),(r,c)))
            energies.append(9999.0)
        e_arr = np.array(energies, dtype=np.float64)
        p = np.exp(-state.beta * (e_arr - e_arr.min()))
        p = p / p.sum()
        idx = np.random.choice(len(cand_positions), p=p)
        chosen[rb.id] = cand_positions[idx]

sample_block(block_a)
sample_block(block_b)

# final uniqueness enforcement
target_to_ids: Dict[Tuple[int,int], List[int]] = {}
for rid, (_, pos) in chosen.items():
    target_to_ids.setdefault(pos, []).append(rid)
for pos, ids in list(target_to_ids.items()):
    if len(ids) > 1:
        ids.sort()
        keep = ids[0]
        for rid in ids[1:]:
            rb = next(r for r in state.robots if r.id == rid)
            chosen[rid] = ((0,0), rb.pos)
return chosen

--------------------------- sim step ---------------------------

def step_sim(state: SimState): state.util_h *= UTIL_DECAY state.util_v *= UTIL_DECAY

# dwell progression and mode transitions
for rb in state.robots:
    if rb.mode in ("loading","unloading") and rb.dwell > 0:
        rb.dwell -= 1
        rb.last_move = (0,0)
        if rb.dwell == 0:
            if rb.mode == "loading":
                rb.carrying = True
                shelf, bay = find_drop_target(state, rb.pos)
                rb.drop_shelf = shelf
                rb.goal = bay
                if shelf is not None:
                    state.reserved[shelf] = rb.id
                rb.mode = "to_drop"
                logc(rb, f"r{rb.id}: load done -> carrying; thrml reserved shelf {shelf} via bay {bay}")
            elif rb.mode == "unloading":
                rb.carrying = False
                if rb.drop_shelf is not None:
                    sr, sc = rb.drop_shelf
                    state.shelves_full[sr, sc] = True
                    if rb.drop_shelf in state.reserved:
                        del state.reserved[rb.drop_shelf]
                rb.drop_shelf = None
                rb.goal = nearest_pick(state.picks, rb.pos)
                rb.mode = "to_pick"
                logc(rb, f"r{rb.id}: unload done -> thrml set next goal to nearest pick {rb.goal}")

proposals = propose_moves(state)

# apply moves with collision resolution
target_to_ids: Dict[Tuple[int,int], List[int]] = {}
for rid, (_, pos) in proposals.items():
    target_to_ids.setdefault(pos, []).append(rid)

for pos, ids in target_to_ids.items():
    ids.sort()
    winner = ids[0]
    losers = ids[1:]
    for rb in state.robots:
        if rb.id == winner:
            v_idx, h_idx = edge_indices(rb.pos, pos)
            if v_idx is not None:
                state.util_v[v_idx] += 1.0
            if h_idx is not None:
                state.util_h[h_idx] += 1.0
            rb.last_move = (pos[0]-rb.pos[0], pos[1]-rb.pos[1])
            rb.pos = pos
        elif rb.id in losers:
            v_idx, h_idx = edge_indices(rb.pos, pos)
            if v_idx is not None:
                state.util_v[v_idx] += 0.3
            if h_idx is not None:
                state.util_h[h_idx] += 0.3

# arrivals
for rb in state.robots:
    if rb.at_goal():
        if rb.mode == "to_pick":
            rb.mode = "loading"
            rb.dwell = LOAD_DWELL
            logc(rb, f"r{rb.id}: arrived pick {rb.pos} -> thrml scheduled LOADING for {LOAD_DWELL}f")
        elif rb.mode == "to_drop" and rb.drop_shelf is not None:
            sr, sc = rb.drop_shelf
            # if shelf got filled or reservation hijacked, retarget
            if state.shelves_full[sr, sc] and state.reserved.get((sr, sc)) != rb.id:
                new_shelf, new_bay = find_drop_target(state, rb.pos)
                rb.drop_shelf = new_shelf
                rb.goal = new_bay
                if new_shelf is not None:
                    state.reserved[new_shelf] = rb.id
                logc(rb, f"r{rb.id}: target shelf full -> thrml retarget {new_shelf} via {new_bay}")
            elif manhattan(rb.pos, (sr, sc)) == 1:
                if (state.reserved.get((sr, sc)) in (None, rb.id)) and not state.shelves_full[sr, sc]:
                    rb.mode = "unloading"
                    rb.dwell = UNLOAD_DWELL
                    logc(rb, f"r{rb.id}: at shelf {rb.drop_shelf} -> thrml scheduled UNLOADING {UNLOAD_DWELL}f")
                else:
                    new_shelf, new_bay = find_drop_target(state, rb.pos)
                    rb.drop_shelf = new_shelf
                    rb.goal = new_bay
                    if new_shelf is not None:
                        state.reserved[new_shelf] = rb.id
                    logc(rb, f"r{rb.id}: contention -> thrml retarget {new_shelf} via {new_bay}")

--------------------------- rendering ---------------------------

def lerp(a: int, b: int, t: float) -> int: return int(a + (b-a)*t)

def util_to_color(v: float) -> Tuple[int,int,int]: v = max(0.0, min(1.0, v)) r = lerp(BG[0], HEAT[0], v) g = lerp(BG[1], HEAT[1], v) b = lerp(BG[2], HEAT[2], v) return (r,g,b)

def draw(state: SimState, screen, show_heat: bool, clock: pygame.time.Clock, H: int, W: int, panel_w: int): obs = state.obs h, w = obs.shape screen.fill(BG)

grid_w_px = W*(CELL+MARGIN)+MARGIN

# cells
for r in range(h):
    for c in range(w):
        x = c * (CELL + MARGIN) + MARGIN
        y = r * (CELL + MARGIN) + MARGIN
        rect = pygame.Rect(x, y, CELL, CELL)
        color = GRID
        if obs[r,c]:
            color = SHELF_FULL if state.shelves_full[r,c] else SHELF
        pygame.draw.rect(screen, color, rect, border_radius=4)
        if obs[r,c] and state.shelves_full[r,c]:
            pygame.draw.rect(screen, CARGO, (x+CELL//2-4, y+CELL//2-4, 8, 8))

# heat overlay
if show_heat:
    for r in range(h):
        for c in range(w-1):
            v = min(1.0, state.util_h[r,c]/2.0)
            if v < 0.02: continue
            x = c*(CELL+MARGIN)+MARGIN + CELL
            y = r*(CELL+MARGIN)+MARGIN + CELL//2-5
            pygame.draw.rect(screen, util_to_color(v), (x, y, MARGIN+2, 10), border_radius=2)
    for r in range(h-1):
        for c in range(w):
            v = min(1.0, state.util_v[r,c]/2.0)
            if v < 0.02: continue
            x = c*(CELL+MARGIN)+MARGIN + CELL//2-5
            y = r*(CELL+MARGIN)+MARGIN + CELL
            pygame.draw.rect(screen, util_to_color(v), (x, y, 10, MARGIN+2), border_radius=2)

# pick stations
for g in state.picks:
    gr, gc = g
    x = gc*(CELL+MARGIN)+MARGIN
    y = gr*(CELL+MARGIN)+MARGIN
    pygame.draw.rect(screen, GOAL, (x+6, y+6, CELL-12, CELL-12), width=3, border_radius=6)

# robots
for rb in state.robots:
    r, c = rb.pos
    x = c*(CELL+MARGIN)+MARGIN
    y = r*(CELL+MARGIN)+MARGIN
    body = pygame.Rect(x+4, y+4, CELL-8, CELL-8)
    pygame.draw.rect(screen, rb.color, body, border_radius=12)
    if rb.carrying:
        pygame.draw.rect(screen, CARGO, (x+CELL//2-5, y+4, 10, 10), border_radius=2)
    if rb.mode == "loading" and rb.dwell > 0:
        t = min(1.0, (LOAD_DWELL - rb.dwell)/LOAD_DWELL)
        pygame.draw.rect(screen, CARGO, (x+6, y+CELL-10, int((CELL-12)*t), 4))
    if rb.mode == "unloading" and rb.dwell > 0:
        t = min(1.0, (UNLOAD_DWELL - rb.dwell)/UNLOAD_DWELL)
        pygame.draw.rect(screen, CARGO, (x+6, y+CELL-10, int((CELL-12)*t), 4))

# hud
font = pygame.font.SysFont("consolas", 16)
txt = f"beta={state.beta:.2f}  cong_w={state.w_cong:.1f}  robots={len(state.robots)}  fps~{int(clock.get_fps())}"
surf = font.render(txt, True, TXT)
screen.blit(surf, (8, 4))

# right-side log panel
panel_x = grid_w_px
pygame.draw.rect(screen, (28,28,34), (panel_x, 0, screen.get_width()-panel_x, screen.get_height()))
title = font.render("event log (thrml-style):", True, (180,200,255))
screen.blit(title, (panel_x+8, 8))
font_small = pygame.font.SysFont("consolas", 7)
lines = list(LOG)[-42:]
y = 32
for text, color in lines:
    surf = font_small.render(text, True, color)
    screen.blit(surf, (panel_x+8, y))
    y += 12

--------------------------- main loop ---------------------------

def random_blockage(state: SimState) -> None: rb = random.choice(state.robots) r, c = rb.pos for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: nr, nc = r+dr, c+dc if in_bounds(state.obs, nr, nc) and state.obs[nr, nc] == 0: state.obs[nr, nc] = 1 state.shelves_full[nr, nc] = True log(f"env: blockage at {(nr,nc)} -> shelf full; thrml will reroute flows") return

def spawn_near_picks(obs: np.ndarray, picks: List[Tuple[int,int]], want_per_pick: int) -> List[Tuple[int,int]]: placed: List[Tuple[int,int]] = [] taken = set() for g in picks: cells = bfs_free_cells(obs, g, limit=want_per_pick*8) cells = [c for c in cells if c != g]  # never on pick for cell in cells: if cell not in taken: placed.append(cell) taken.add(cell) if sum(1 for p in placed if manhattan(p, g) <= 6) >= want_per_pick: break return placed

def build_sim(preset: int = 0, n_robots: Optional[int] = None) -> SimState: obs, nominal_starts, picks = preset_map(preset)

# spawn near picks, never on pick station
spawn_points = [c for c in spawn_near_picks(obs, picks, ROBOTS_PER_PICK) if c not in picks]
if n_robots is None:
    n_robots = min(len(spawn_points), MAX_ROBOTS)
else:
    n_robots = min(n_robots, len(spawn_points), MAX_ROBOTS)

robots: List[Robot] = []
for i in range(n_robots):
    s = spawn_points[i]
    g = nearest_pick(picks, s)
    robots.append(Robot(i, s, g, ROBOT_COLORS[i % len(ROBOT_COLORS)], mode="to_pick"))
    logc(robots[-1], f"spawn r{i} at {s} -> thrml set goal pick {g}")

util_h = np.zeros((obs.shape[0], obs.shape[1]-1), dtype=np.float32)
util_v = np.zeros((obs.shape[0]-1, obs.shape[1]), dtype=np.float32)
shelves_full = np.zeros_like(obs, dtype=bool)
return SimState(obs=obs, shelves_full=shelves_full, robots=robots, picks=picks,
                util_h=util_h, util_v=util_v, beta=DEFAULT_BETA, w_cong=DEFAULT_W_CONG,
                reserved={})

if name == "main": pygame.init() random.seed(SEED) np.random.seed(SEED)

# default to map 4 (we call it preset 0 here for simplicity)
preset = 0
state = build_sim(preset)
H, W = state.obs.shape

panel_w = 360
screen = pygame.display.set_mode((W*(CELL+MARGIN)+MARGIN + panel_w, H*(CELL+MARGIN)+MARGIN))
pygame.display.set_caption("multi-robot warehouse (pick -> carry -> putaway) -- map 4 default")
clock = pygame.time.Clock()

log("init: map=4 preset, thrml active; building sim and placing robots near picks (no spawn on stations)")

paused = False
show_heat = True

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                state = build_sim(preset)
                H, W = state.obs.shape
                screen = pygame.display.set_mode((W*(CELL+MARGIN)+MARGIN + panel_w, H*(CELL+MARGIN)+MARGIN))
                log("reset: rebuilt sim on map 4 with clustered spawns; thrml params preserved")
            elif event.key == pygame.K_b:
                random_blockage(state)
            elif event.key == pygame.K_g:
                show_heat = not show_heat
            elif event.key == pygame.K_LEFTBRACKET:
                state.beta = max(0.2, state.beta - 0.1)
            elif event.key == pygame.K_RIGHTBRACKET:
                state.beta = min(5.0, state.beta + 0.1)
            elif event.key == pygame.K_COMMA:
                state.w_cong = max(0.0, state.w_cong - 0.2)
            elif event.key == pygame.K_PERIOD:
                state.w_cong = state.w_cong + 0
