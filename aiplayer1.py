import random
import numpy as np
import time
import math
import pickle
import os
from collections import defaultdict

# First huge shoutout to my girlfriend for all the support (She is a seconde Year DSAI student)!
# This AI is called Mahoraga (meaning: "Demon God") after the anime Jujutsu Kaisen
# why? Because I love the anime and the character Mahoraga is very adaptive, just like this AI.
# this AI adepts just like Mahoraga to the opponent and learns from its mistakes.
# after 5 games it will start to play better and better.

# ---------- Config ----------
cookies = {
    "center": 5,
    "two_in_a_row": 2,
    "three_in_a_row": 10,
    "punish_repeat": -50,
    "block_opponent": 8,
    "counter_center": 3
}

ROW_COUNT = 6
COLUMN_COUNT = 7

# Time limit per move (seconds)
TIME_LIMIT = 2.0

# File to persist experience / pattern DB
MEMORY_FILE = "ai_memory.pkl"
PATTERN_DB_FILE = "pattern_db.pkl"

# Stochastic params
EPSILON_RANDOM_MOVE = 0.03  # small chance to pick fully random move
GAUSSIAN_NOISE_STD = 1e-3  # tiny noise to diversify ties

# Aspiration margin
ASPIRATION_MARGIN = 25  # +/- margin around previous score for aspiration window

# ---------- Utilities ----------
def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_valid_locations(board):
    return [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
    return None

def check_center_domination(board, player_id):
    """
    Check if opponent dominates center and return counter moves.
    Return list of preferred columns to counter center strategy.
    """
    center_col = COLUMN_COUNT // 2
    opponent_id = 3 - player_id
    
    # Check if opponent dominates center
    center_count = sum(1 for r in range(ROW_COUNT) if board[r][center_col] == opponent_id)
    
    if center_count >= 2:  # opponent dominates center
        # Prioritize columns next to center
        return [center_col-1, center_col+1, center_col-2, center_col+2]
    return []

def get_punished_moves(board, player_id):
    """
    Return moves that should be punished (losing moves from experience).
    """
    key = board.tobytes()
    if key in experience:
        # Check if last move was the losing move (negative score)
        if experience[key]["last_score"] < -1000:
            return [experience[key]["move"]]  
    return []


def drop_piece(board, row, col, piece):
    board[row][col] = piece

def winning_move(board, piece):
    # Horizontal
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if all(board[r][c+i] == piece for i in range(4)):
                return True
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if all(board[r+i][c] == piece for i in range(4)):
                return True
    # Positive diagonal
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if all(board[r+i][c+i] == piece for i in range(4)):
                return True
    # Negative diagonal
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if all(board[r-i][c+i] == piece for i in range(4)):
                return True
    return False

# ---------- Pattern DB and memory ----------
def build_pattern_db():
    """
    Precompute scores for all 4-length windows (0,1,2 values) to speed evaluation.
    There are 3^4 = 81 patterns.
    """
    if os.path.exists(PATTERN_DB_FILE):
        with open(PATTERN_DB_FILE, "rb") as f:
            return pickle.load(f)
    db = {}
    from itertools import product
    for pattern in product((0,1,2), repeat=4):
        window = list(pattern)
        score = 0
        # replicate evaluate_window logic
        if window.count(1) == 4:
            score += 100
        elif window.count(1) == 3 and window.count(0) == 1:
            score += cookies["three_in_a_row"]
        elif window.count(1) == 2 and window.count(0) == 2:
            score += cookies["two_in_a_row"]
        if window.count(2) == 3 and window.count(0) == 1:
            score -= cookies["block_opponent"]
        db[tuple(window)] = score
    with open(PATTERN_DB_FILE, "wb") as f:
        pickle.dump(db, f)
    return db

pattern_db = build_pattern_db()

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "rb") as f:
            return pickle.load(f)
    return defaultdict(lambda: {"visits":0, "last_score":0, "move":None})

def save_memory(mem):
    with open(MEMORY_FILE, "wb") as f:
        pickle.dump(dict(mem), f)

experience = load_memory()

# ---------- Evaluation ----------
def evaluate_window(window):
    """
    Use pattern_db for quick lookup.
    """
    key = tuple(int(x) for x in window)
    return pattern_db.get(key, 0)

def eval_board(board):
    score = 0
    # Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
    center_count = center_array.count(1)
    score += center_count * cookies["center"]
    
    # Counter center domination bonus
    opponent_center_count = center_array.count(2)
    if opponent_center_count >= 2:
        # Bonus for countering center strategy
        score += cookies["counter_center"] * (opponent_center_count - 1)

    # Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r,:])]
        for c in range(COLUMN_COUNT-3):
            window = row_array[c:c+4]
            score += evaluate_window(window)
    # Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT-3):
            window = col_array[r:r+4]
            score += evaluate_window(window)

    # Score positive sloped diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(4)]
            score += evaluate_window(window)

    # Score negative sloped diagonal
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(4)]
            score += evaluate_window(window)

    return score

def is_terminal_node(board):
    return winning_move(board, 1) or winning_move(board, 2) or len(get_valid_locations(board)) == 0

# ---------- Threat detection ----------
def immediate_win_or_block(board, piece):
    """
    Check for immediate winning move for 'piece' (1 or 2).
    If the current player (piece) has an immediate winning move, return that col.
    Otherwise, if opponent has immediate winning move, return the blocking col.
    Return None if none found.
    Priority: Win for AI > Block opponent's win.
    """
    valid_locations = get_valid_locations(board)
    # Check self win
    for col in valid_locations:
        row = get_next_open_row(board, col)
        if row is None: continue
        b_copy = board.copy()
        drop_piece(b_copy, row, col, piece)
        if winning_move(b_copy, piece):
            return col
    # Check opponent
    opp = 1 if piece == 2 else 2
    for col in valid_locations:
        row = get_next_open_row(board, col)
        if row is None: continue
        b_copy = board.copy()
        drop_piece(b_copy, row, col, opp)
        if winning_move(b_copy, opp):
            return col
    return None

# ---------- Minimax with alpha-beta, move ordering, aspiration ----------
def minimax_ab(board, depth, alpha, beta, maximizingPlayer, start_time, time_limit, move_order=None):
    """
    Regular alphabeta with move ordering and time checks.
    Returns (best_col, value, completed_flag)
    completed_flag = True if this search completed within time limit, else False.
    """
    # Time check
    if time.time() - start_time > time_limit:
        return None, 0, False

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, 1):
                return (None, 1e12, True)
            elif winning_move(board, 2):
                return (None, -1e12, True)
            else: # draw
                return (None, 0, True)
        else:
            return (None, eval_board(board), True)

    if move_order is None:
        # default order: as is
        move_order = valid_locations

    if maximizingPlayer:
        value = -math.inf
        best_cols = []
        for col in move_order:
            # time check
            if time.time() - start_time > time_limit:
                return None, value, False
            row = get_next_open_row(board, col)
            if row is None:
                continue
            b_copy = board.copy()
            drop_piece(b_copy, row, col, 1)
            _, new_score, completed = minimax_ab(b_copy, depth-1, alpha, beta, False, start_time, time_limit, move_order=None)
            if not completed:
                return None, value, False
            # Add tiny noise to diversify
            new_score = new_score + random.gauss(0, GAUSSIAN_NOISE_STD)
            if new_score > value + 1e-9:
                value = new_score
                best_cols = [col]
            elif abs(new_score - value) <= 1e-9:
                best_cols.append(col)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        if not best_cols:
            return random.choice(valid_locations), value, True
        return random.choice(best_cols), value, True
    else:
        value = math.inf
        best_cols = []
        for col in move_order:
            if time.time() - start_time > time_limit:
                return None, value, False
            row = get_next_open_row(board, col)
            if row is None:
                continue
            b_copy = board.copy()
            drop_piece(b_copy, row, col, 2)
            _, new_score, completed = minimax_ab(b_copy, depth-1, alpha, beta, True, start_time, time_limit, move_order=None)
            if not completed:
                return None, value, False
            new_score = new_score + random.gauss(0, GAUSSIAN_NOISE_STD)
            if new_score < value - 1e-9:
                value = new_score
                best_cols = [col]
            elif abs(new_score - value) <= 1e-9:
                best_cols.append(col)
            beta = min(beta, value)
            if alpha >= beta:
                break
        if not best_cols:
            return random.choice(valid_locations), value, True
        return random.choice(best_cols), value, True

# ---------- Helper for move ordering via shallow search ----------
def shallow_scores_for_moves(board, shallow_depth=3, time_limit=1.0):
    """
    Do a small-depth search for each candidate move to produce an ordering.
    Returns list of (col, score) sorted descending (best first for maximizing player).
    """
    start_time = time.time()
    valid_locations = get_valid_locations(board)
    results = []
    for col in valid_locations:
        if time.time() - start_time > time_limit:
            break
        row = get_next_open_row(board, col)
        if row is None:
            continue
        b_copy = board.copy()
        drop_piece(b_copy, row, col, 1)  # AI plays
        # single minimax call at shallow depth (no move ordering needed)
        _, score, completed = minimax_ab(b_copy, shallow_depth-1, -math.inf, math.inf, False, start_time, time_limit, move_order=None)
        if not completed:
            # if not completed, fallback to eval
            score = eval_board(b_copy)
        results.append((col, score))
    # sort by descending score (best first)
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ---------- Iterative deepening with aspiration ----------
def find_best_move_iterative(board, max_time=TIME_LIMIT, max_depth=6, start_depth=1):
    """
    Iterative deepening driver. Keeps the best move from the last fully completed depth.
    Uses move ordering and aspiration windows.
    """
    start_time = time.time()
    best_move = None
    best_score = -math.inf
    prev_score = 0

    # immediate forced moves first (win/block)
    forced = immediate_win_or_block(board, 1)
    if forced is not None:
        save_board_experience(board, forced, 1, reason="forced_win_or_block")
        return forced

    valid_locations = get_valid_locations(board)
    
    # Check for punished moves from experience
    punished_moves = get_punished_moves(board, 1)
    valid_locations = [col for col in valid_locations if col not in punished_moves]
    
    if not valid_locations:
        # Fallback if all moves are punished
        valid_locations = get_valid_locations(board)

    # small epsilon randomization
    if random.random() < EPSILON_RANDOM_MOVE and valid_locations:
        rand_col = random.choice(valid_locations)
        save_board_experience(board, rand_col, 1, reason="epsilon_random")
        return rand_col

    # Counter center domination strategy
    center_counter_moves = check_center_domination(board, 1)
    if center_counter_moves:
        # add center counter moves first in ordering
        valid_center_moves = [col for col in center_counter_moves if col in valid_locations]
        if valid_center_moves:
            move_order = valid_center_moves + [col for col in valid_locations if col not in valid_center_moves]
        else:
            move_order = valid_locations
    else:
        # Normal move ordering
        shallow = shallow_scores_for_moves(board, shallow_depth=3, time_limit=min(0.8, max_time/4))
        move_order = [col for col, _ in shallow]
        remaining = [c for c in valid_locations if c not in move_order]
        move_order.extend(remaining)

    completed_depth = 0
    for depth in range(start_depth, max_depth+1):
        if time.time() - start_time > max_time:
            break

        # aspiration window based on prev_score
        if completed_depth > 0:
            alpha = prev_score - ASPIRATION_MARGIN
            beta = prev_score + ASPIRATION_MARGIN
            # run aspiration search
            col, score, completed = minimax_ab(board, depth, alpha, beta, True, start_time, max_time - (time.time() - start_time), move_order=move_order)
            if not completed:
                # timeout — exit returning last best
                break
            if score <= alpha or score >= beta:
                # outside aspiration window -> re-search with full window
                col, score, completed = minimax_ab(board, depth, -math.inf, math.inf, True, start_time, max_time - (time.time() - start_time), move_order=move_order)
                if not completed:
                    break
        else:
            # first completed depth search with full window
            col, score, completed = minimax_ab(board, depth, -math.inf, math.inf, True, start_time, max_time - (time.time() - start_time), move_order=move_order)

        if not completed:
            # didn't finish this depth in time; stop and return last completed best
            break

        # On completion, store results
        best_move = col
        best_score = score
        prev_score = score
        completed_depth = depth

        # Update move ordering with this deeper result: put best move first
        if best_move in move_order:
            move_order.remove(best_move)
        move_order.insert(0, best_move)

        # small early exit if winning move found
        if abs(best_score) > 9e11:
            break

    # If multiple good moves, choose randomly among top ones (tie-break)
    if best_move is None and valid_locations:
        best_move = random.choice(valid_locations)

    # Save experience about this board
    save_board_experience(board, best_move, best_score, reason=f"depth_{completed_depth}")

    return best_move

# ---------- Experience storage ----------
def save_board_experience(board, move, score, reason="move"):
    """
    Save board bytes -> stats: visits, last_score, preferred_move
    """
    key = board.tobytes()
    rec = experience.get(key, {"visits":0, "last_score":0, "move":None})
    rec["visits"] = rec.get("visits",0) + 1
    rec["last_score"] = score
    rec["move"] = move
    rec["reason"] = reason
    
    if score < -1000:
        rec["punished"] = True
    
    experience[key] = rec
    try:
        save_memory(experience)
    except Exception:
        pass

# ---------- Public API ----------
def aiplayer1(board, time_limit=TIME_LIMIT, max_depth=6):
    """
    Board: numpy array shape (6,7) with 0 empty, 1 AI, 2 opponent.
    Returns chosen column to play.
    """
    # Ensure board is numpy array copy
    b = np.array(board, copy=True)
    col = find_best_move_iterative(b, max_time=time_limit, max_depth=max_depth, start_depth=1)
    return col

# ---------- usage ----------
if __name__ == "__main__":
    # quick sanity test - empty board
    board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=int)
    move = aiplayer1(board, time_limit=2.0, max_depth=5)
    print("Chosen move:", move)