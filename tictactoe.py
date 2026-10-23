from __future__ import annotations

import json
import os
import random
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

PLAYER_X: str = "X"
PLAYER_O: str = "O"
EMPTY: str = " "

MODE_TWO_PLAYER: str = "pvp"
MODE_VS_COMPUTER: str = "pvc"

DIFFICULTY_EASY: str = "easy"
DIFFICULTY_MEDIUM: str = "medium"
DIFFICULTY_HARD: str = "hard"

WINNING_COMBINATIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),  # Row 1
    (3, 4, 5),  # Row 2
    (6, 7, 8),  # Row 3
    (0, 3, 6),  # Column 1
    (1, 4, 7),  # Column 2
    (2, 5, 8),  # Column 3
    (0, 4, 8),  # Diagonal 1
    (2, 4, 6),  # Diagonal 2
)

SCORES_FILE: str = "scores.json"
DEFAULT_HOST: str = "127.0.0.1"
DEFAULT_PORT: int = 5000


# ============================================================================
# PURE LOGIC MODEL & MINIMAX AI
# ============================================================================

class TicTacToeGame:
    """Pure-logic engine for a 3x3 Tic-Tac-Toe game.

    Maintains the board state, move history, turn switching, and victory
    checks without any GUI or web framework dependencies.
    """

    def __init__(self, starting_player: str = PLAYER_X) -> None:
        """Initialize the game state.

        Args:
            starting_player: The mark ('X' or 'O') that takes the first move.
        """
        self.starting_player: str = starting_player
        self.current_player: str = starting_player
        self.board: List[str] = [EMPTY] * 9
        self.move_history: List[int] = []
        self.winner: Optional[str] = None
        self.winning_line: Optional[Tuple[int, int, int]] = None

    def reset_board(self, starting_player: Optional[str] = None) -> None:
        """Reset the board for a new round.

        Args:
            starting_player: If provided, sets the starting player; otherwise
                reuses the existing starting_player attribute.
        """
        if starting_player is not None:
            self.starting_player = starting_player
        self.current_player = self.starting_player
        self.board = [EMPTY] * 9
        self.move_history.clear()
        self.winner = None
        self.winning_line = None

    def available_moves(self) -> List[int]:
        """Return indices of all currently empty cells."""
        return [i for i, cell in enumerate(self.board) if cell == EMPTY]

    def make_move(self, index: int) -> bool:
        """Attempt to play the current player's mark at the given index.

        Args:
            index: Zero-based cell index (0 to 8).

        Returns:
            True if the move was successfully placed; False if invalid or if
            the game has already concluded.
        """
        if self.is_game_over() or not (0 <= index < 9) or self.board[index] != EMPTY:
            return False

        self.board[index] = self.current_player
        self.move_history.append(index)

        # Check if this move completes a winning combination
        result = self.check_winner()
        if result is not None:
            self.winner, self.winning_line = result
        else:
            # Switch player turn if the game is still undecided
            self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X

        return True

    def undo_move(self) -> Optional[int]:
        """Revert the most recent move on the board.

        Returns:
            The cell index reverted, or None if no moves have been made.
        """
        if not self.move_history:
            return None

        last_move = self.move_history.pop()
        player_who_moved = self.board[last_move]
        self.board[last_move] = EMPTY

        # Reset victory status if previously finished
        self.winner = None
        self.winning_line = None

        # Restore turn to the player whose move was undone
        self.current_player = player_who_moved
        return last_move

    def check_winner(self) -> Optional[Tuple[str, Tuple[int, int, int]]]:
        """Examine the current board against all 8 winning combinations.

        Returns:
            A tuple of (winning_mark, winning_combo) if a winner exists;
            otherwise None.
        """
        for combo in WINNING_COMBINATIONS:
            a, b, c = combo
            if self.board[a] != EMPTY and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a], combo
        return None

    def is_draw(self) -> bool:
        """Check if all cells are filled with no victor."""
        return (self.winner is None) and (EMPTY not in self.board)

    def is_game_over(self) -> bool:
        """Check if the game has ended via either victory or draw."""
        return self.winner is not None or self.is_draw()

    @property
    def move_count(self) -> int:
        """Return the number of moves made in the active round."""
        return len(self.move_history)

    def to_dict(self) -> Dict[str, Any]:
        """Export game state as a serializable dictionary."""
        return {
            "board": self.board,
            "currentPlayer": self.current_player,
            "startingPlayer": self.starting_player,
            "winner": self.winner,
            "winningLine": list(self.winning_line) if self.winning_line else None,
            "isDraw": self.is_draw(),
            "isGameOver": self.is_game_over(),
            "moveCount": self.move_count,
        }

    # ------------------------------------------------------------------------
    # AI OPPONENT STRATEGIES
    # ------------------------------------------------------------------------

    def computer_move(self, difficulty: str) -> Optional[int]:
        """Compute the computer's next cell selection based on difficulty.

        Args:
            difficulty: 'easy', 'medium', or 'hard'.

        Returns:
            Best cell index for the computer (O), or None if no moves exist.
        """
        available = self.available_moves()
        if not available:
            return None

        if difficulty == DIFFICULTY_EASY:
            return random.choice(available)

        if difficulty == DIFFICULTY_MEDIUM:
            return self._heuristic_move(available)

        # Hard mode: Unbeatable Minimax algorithm
        return self._best_minimax_move()

    def _heuristic_move(self, available: List[int]) -> int:
        """Medium AI: win if possible, block opponent win, else center or random."""
        comp = self.current_player
        opponent = PLAYER_X if comp == PLAYER_O else PLAYER_O

        # 1. Take immediate winning move if available
        for move in available:
            self.board[move] = comp
            if self.check_winner() is not None:
                self.board[move] = EMPTY
                return move
            self.board[move] = EMPTY

        # 2. Block immediate opponent winning move
        for move in available:
            self.board[move] = opponent
            if self.check_winner() is not None:
                self.board[move] = EMPTY
                return move
            self.board[move] = EMPTY

        # 3. Take center if free
        if 4 in available:
            return 4

        # 4. Otherwise play randomly
        return random.choice(available)

    def _best_minimax_move(self) -> int:
        """Determine optimal move using the Minimax algorithm with alpha-beta pruning.

        Payoff evaluation:
            +10 for Computer (O) victory
            -10 for Human (X) victory
              0 for a Draw
        Depth is subtracted from wins / added to losses to prefer quicker wins
        and prolonged defenses.
        """
        available = self.available_moves()
        if len(available) == 9:
            return random.choice([0, 2, 4, 6, 8])
        if len(available) == 1:
            return available[0]

        best_score = -float("inf")
        best_move = available[0]
        alpha = -float("inf")
        beta = float("inf")

        for move in available:
            self.board[move] = PLAYER_O
            score = self.minimax(depth=0, is_maximizing=False, alpha=alpha, beta=beta)
            self.board[move] = EMPTY

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        return best_move

    def minimax(self, depth: int, is_maximizing: bool, alpha: float, beta: float) -> int:
        """Recursive Minimax search with alpha-beta pruning.

        Args:
            depth: Number of plies explored from root.
            is_maximizing: True for O (maximizer), False for X (minimizer).
            alpha: Best score maximized so far.
            beta: Best score minimized so far.

        Returns:
            The evaluated heuristic payoff for the explored state.
        """
        win_info = self.check_winner()
        if win_info is not None:
            winner_mark, _ = win_info
            if winner_mark == PLAYER_O:
                return 10 - depth
            else:
                return depth - 10

        available = self.available_moves()
        if not available:
            return 0

        if is_maximizing:
            max_eval = -float("inf")
            for move in available:
                self.board[move] = PLAYER_O
                evaluation = self.minimax(depth + 1, False, alpha, beta)
                self.board[move] = EMPTY
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break
            return int(max_eval)
        else:
            min_eval = float("inf")
            for move in available:
                self.board[move] = PLAYER_X
                evaluation = self.minimax(depth + 1, True, alpha, beta)
                self.board[move] = EMPTY
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break
            return int(min_eval)


# ============================================================================
# SCOREBOARD & PERSISTENCE
# ============================================================================

class ScoreManager:
    """Manages persistent score data safely using scores.json."""

    def __init__(self, filepath: str = SCORES_FILE) -> None:
        self.filepath = filepath
        self.scores: Dict[str, int] = {PLAYER_X: 0, PLAYER_O: 0, "ties": 0}
        self.load()

    def record_result(self, winner: Optional[str]) -> None:
        """Update scores following a round conclusion."""
        if winner == PLAYER_X:
            self.scores[PLAYER_X] += 1
        elif winner == PLAYER_O:
            self.scores[PLAYER_O] += 1
        else:
            self.scores["ties"] += 1
        self.save()

    def reset(self) -> None:
        """Reset scores to zero."""
        self.scores = {PLAYER_X: 0, PLAYER_O: 0, "ties": 0}
        self.save()

    def load(self) -> None:
        """Load scores from disk with safe error fallback."""
        if not os.path.exists(self.filepath):
            return
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.scores[PLAYER_X] = int(data.get(PLAYER_X, 0))
                    self.scores[PLAYER_O] = int(data.get(PLAYER_O, 0))
                    self.scores["ties"] = int(data.get("ties", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            self.scores = {PLAYER_X: 0, PLAYER_O: 0, "ties": 0}

    def save(self) -> None:
        """Save scores to disk with safe error handling."""
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.scores, f, indent=2)
        except OSError:
            pass


# ============================================================================
# EMBEDDED WEB FRONTEND (HTML, CSS, JS)
# ============================================================================

HTML_PAGE: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="description" content="A modern Python-powered Web Tic-Tac-Toe application with Unbeatable Minimax AI.">
  <title>Tic-Tac-Toe | Python Web Application</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0b0f19;
      --bg-surface: rgba(22, 30, 49, 0.75);
      --bg-surface-hover: rgba(30, 41, 67, 0.85);
      --bg-cell: rgba(26, 35, 58, 0.65);
      --bg-cell-hover: rgba(38, 51, 84, 0.85);
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-bright: rgba(255, 255, 255, 0.16);
      --color-x: #38bdf8;
      --color-x-glow: rgba(56, 189, 248, 0.35);
      --color-o: #fb7185;
      --color-o-glow: rgba(251, 113, 133, 0.35);
      --color-win: #34d399;
      --color-win-glow: rgba(52, 211, 153, 0.45);
      --color-win-bg: rgba(52, 211, 153, 0.15);
      --text-primary: #f8fafc;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --font-main: 'Outfit', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      --radius-sm: 8px;
      --radius-md: 14px;
      --radius-lg: 20px;
      --radius-xl: 28px;
      --shadow-card: 0 16px 36px -8px rgba(0, 0, 0, 0.5), 0 0 0 1px var(--border-subtle);
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: var(--font-main);
      background-color: var(--bg-dark);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 1.5rem 1rem;
      position: relative;
      overflow-x: hidden;
    }
    .ambient-glow {
      position: fixed;
      border-radius: 50%;
      filter: blur(100px);
      opacity: 0.25;
      pointer-events: none;
      z-index: 0;
    }
    .glow-1 { width: 450px; height: 450px; background: radial-gradient(circle, #38bdf8, transparent 70%); top: -100px; left: -100px; }
    .glow-2 { width: 500px; height: 500px; background: radial-gradient(circle, #e11d48, transparent 70%); bottom: -120px; right: -120px; }
    .confetti-canvas { position: fixed; inset: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 99; }
    .app-layout { position: relative; z-index: 1; width: 100%; max-width: 460px; display: flex; flex-direction: column; gap: 1.1rem; }
    .app-header { text-align: center; display: flex; flex-direction: column; align-items: center; gap: 0.35rem; }
    .brand-badge {
      font-family: var(--font-mono); font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
      padding: 0.25rem 0.75rem; border-radius: 9999px; background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--border-bright); color: var(--text-secondary);
    }
    .app-title {
      font-size: 2.1rem; font-weight: 800; letter-spacing: -0.03em;
      background: linear-gradient(135deg, #ffffff 40%, #cbd5e1 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .app-subtitle { font-size: 0.88rem; color: var(--text-secondary); }
    .controls-card {
      background: var(--bg-surface); backdrop-filter: blur(16px);
      border-radius: var(--radius-lg); border: 1px solid var(--border-subtle);
      box-shadow: var(--shadow-card); padding: 0.85rem 1rem;
      display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; flex-wrap: wrap;
    }
    .control-group { display: flex; flex-direction: column; gap: 0.25rem; flex: 1; min-width: 130px; }
    .control-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }
    .custom-select {
      width: 100%; background: rgba(11, 15, 25, 0.6); border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm); color: var(--text-primary); font-family: var(--font-main);
      font-size: 0.84rem; font-weight: 600; padding: 0.45rem 0.65rem; cursor: pointer; outline: none;
    }
    .control-actions { display: flex; gap: 0.4rem; align-items: flex-end; }
    .icon-btn {
      background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border-subtle);
      color: var(--text-primary); width: 36px; height: 36px; border-radius: var(--radius-sm);
      display: flex; align-items: center; justify-content: center; cursor: pointer;
    }
    .icon-btn:hover { background: rgba(255, 255, 255, 0.12); }
    .status-bar { display: flex; align-items: center; justify-content: space-between; padding: 0.2rem 0.25rem; }
    .status-indicator { display: flex; align-items: center; gap: 0.6rem; }
    .status-pulse { width: 10px; height: 10px; border-radius: 50%; background: var(--color-x); box-shadow: 0 0 10px var(--color-x); }
    .status-text { font-size: 1.05rem; font-weight: 700; color: var(--text-primary); }
    .meta-stats { display: flex; gap: 0.4rem; }
    .stat-pill {
      font-family: var(--font-mono); font-size: 0.74rem; font-weight: 600; padding: 0.25rem 0.55rem;
      border-radius: 6px; background: rgba(255, 255, 255, 0.05); color: var(--text-secondary);
      border: 1px solid var(--border-subtle);
    }
    .board-wrapper { position: relative; width: 100%; }
    .game-board {
      position: relative; display: grid; grid-template-columns: repeat(3, 1fr);
      grid-template-rows: repeat(3, 1fr); gap: 12px; background: var(--bg-surface);
      backdrop-filter: blur(20px); border: 1px solid var(--border-subtle);
      border-radius: var(--radius-xl); box-shadow: var(--shadow-card); padding: 14px;
      aspect-ratio: 1 / 1;
    }
    .cell {
      position: relative; background: var(--bg-cell); border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md); cursor: pointer; display: flex; align-items: center;
      justify-content: center; transition: all 0.15s ease; outline: none; user-select: none;
    }
    .cell:hover:not(.occupied):not(.disabled) { background: var(--bg-cell-hover); transform: translateY(-2px); }
    .cell.occupied { cursor: default; }
    .cell.win-cell { background: var(--color-win-bg) !important; border-color: var(--color-win) !important; box-shadow: 0 0 25px var(--color-win-glow) !important; }
    .mark-x { color: var(--color-x); text-shadow: 0 0 24px var(--color-x-glow); font-size: clamp(2.6rem, 10vw, 3.8rem); font-weight: 800; animation: pop 0.25s cubic-bezier(0.34, 1.56, 0.64, 1); }
    .mark-o { color: var(--color-o); text-shadow: 0 0 24px var(--color-o-glow); font-size: clamp(2.6rem, 10vw, 3.8rem); font-weight: 800; animation: pop 0.25s cubic-bezier(0.34, 1.56, 0.64, 1); }
    .cell.win-cell .mark-x, .cell.win-cell .mark-o { color: var(--color-win) !important; text-shadow: 0 0 25px var(--color-win-glow) !important; }
    @keyframes pop { 0% { transform: scale(0.3); opacity: 0; } 100% { transform: scale(1); opacity: 1; } }
    .mark-ghost { opacity: 0; font-size: clamp(2.4rem, 9vw, 3.4rem); font-weight: 800; pointer-events: none; transition: opacity 0.15s; }
    .cell:hover:not(.occupied):not(.disabled) .mark-ghost { opacity: 0.2; }
    .winning-strike { position: absolute; inset: 14px; pointer-events: none; z-index: 10; opacity: 0; transition: opacity 0.25s; }
    .winning-strike.active { opacity: 1; }
    .winning-strike line { stroke: var(--color-win); stroke-width: 8; stroke-linecap: round; filter: drop-shadow(0 0 10px var(--color-win)); stroke-dasharray: 400; stroke-dashoffset: 400; transition: stroke-dashoffset 0.45s ease-out; }
    .winning-strike.draw line { stroke-dashoffset: 0; }
    .scoreboard-card {
      background: var(--bg-surface); backdrop-filter: blur(16px); border-radius: var(--radius-lg);
      border: 1px solid var(--border-subtle); box-shadow: var(--shadow-card); padding: 0.85rem 1.2rem;
      display: flex; align-items: center; justify-content: space-between;
    }
    .score-item { display: flex; align-items: center; gap: 0.65rem; flex: 1; }
    .score-item.score-ties { justify-content: center; }
    .score-item.score-o { justify-content: flex-end; }
    .score-badge { width: 32px; height: 32px; border-radius: var(--radius-sm); display: flex; align-items: center; justify-content: center; font-weight: 800; }
    .badge-x { background: rgba(56, 189, 248, 0.15); color: var(--color-x); }
    .badge-o { background: rgba(251, 113, 133, 0.15); color: var(--color-o); }
    .badge-tie { background: rgba(255, 255, 255, 0.08); font-size: 0.85rem; }
    .score-info { display: flex; flex-direction: column; }
    .score-label { font-size: 0.72rem; font-weight: 600; color: var(--text-secondary); white-space: nowrap; }
    .score-value { font-family: var(--font-mono); font-size: 1.25rem; font-weight: 700; color: var(--text-primary); }
    .score-divider { width: 1px; height: 28px; background: var(--border-subtle); }
    .action-bar { display: flex; gap: 0.65rem; }
    .action-btn {
      flex: 1; display: flex; align-items: center; justify-content: center; gap: 0.45rem;
      padding: 0.75rem 0.65rem; border-radius: var(--radius-md); font-family: var(--font-main);
      font-size: 0.88rem; font-weight: 700; cursor: pointer; outline: none; border: 1px solid transparent;
      transition: all 0.15s ease;
    }
    .primary-btn { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: #fff; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4); }
    .primary-btn:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55); }
    .secondary-btn { background: var(--bg-surface); border-color: var(--border-subtle); color: var(--text-primary); }
    .secondary-btn:hover:not(:disabled) { background: var(--bg-surface-hover); transform: translateY(-2px); }
    .action-btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .key-hint { font-family: var(--font-mono); font-size: 0.65rem; padding: 0.15rem 0.35rem; border-radius: 4px; background: rgba(0, 0, 0, 0.25); border: 1px solid rgba(255, 255, 255, 0.15); color: rgba(255, 255, 255, 0.75); }
    .app-footer { text-align: center; }
    .shortcuts-legend { font-size: 0.75rem; color: var(--text-muted); }
    .shortcuts-legend kbd { font-family: var(--font-mono); font-size: 0.7rem; padding: 0.1rem 0.3rem; border-radius: 4px; background: rgba(255, 255, 255, 0.08); border: 1px solid var(--border-subtle); color: var(--text-secondary); }
    .modal-overlay { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.75); backdrop-filter: blur(8px); display: flex; align-items: center; justify-content: center; padding: 1.5rem; z-index: 100; transition: opacity 0.25s, visibility 0.25s; }
    .modal-overlay.hidden { opacity: 0; visibility: hidden; pointer-events: none; }
    .modal-card { background: #151d30; border: 1px solid var(--border-bright); border-radius: var(--radius-xl); box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8); padding: 2rem 1.75rem; width: 100%; max-width: 400px; text-align: center; display: flex; flex-direction: column; align-items: center; }
    .modal-icon { font-size: 3.2rem; margin-bottom: 0.5rem; }
    .modal-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 0.35rem; }
    .modal-subtitle { font-size: 0.92rem; color: var(--text-secondary); margin-bottom: 1.5rem; }
    .modal-actions { display: flex; gap: 0.75rem; width: 100%; }
    .help-card { max-width: 480px; text-align: left; align-items: stretch; }
    .help-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
    .close-btn { background: transparent; border: none; color: var(--text-secondary); font-size: 1.5rem; cursor: pointer; }
    .help-content h3 { font-size: 0.95rem; font-weight: 700; color: var(--color-x); margin: 0.8rem 0 0.3rem; }
    .help-content p, .help-content li { font-size: 0.86rem; color: var(--text-secondary); }
    .shortcut-grid { display: grid; grid-template-columns: auto 1fr; gap: 0.4rem 1rem; align-items: center; font-size: 0.84rem; color: var(--text-secondary); background: rgba(0, 0, 0, 0.25); padding: 0.75rem; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); margin-top: 0.5rem; }
    .shortcut-grid kbd { font-family: var(--font-mono); font-size: 0.74rem; padding: 0.15rem 0.4rem; border-radius: 4px; background: rgba(255, 255, 255, 0.1); color: var(--text-primary); border: 1px solid var(--border-bright); }
  </style>
</head>
<body>
  <div class="ambient-glow glow-1"></div>
  <div class="ambient-glow glow-2"></div>
  <canvas id="confetti-canvas" class="confetti-canvas"></canvas>

  <div class="app-layout">
    <header class="app-header">
      <div class="brand-badge">PYTHON WEB ENGINE</div>
      <h1 class="app-title">TIC-TAC-TOE</h1>
      <p class="app-subtitle">Powered by Python & Unbeatable Minimax AI</p>
    </header>

    <section class="controls-card">
      <div class="control-group">
        <label for="mode-select" class="control-label">Game Mode</label>
        <select id="mode-select" class="custom-select">
          <option value="pvc" selected>vs. Computer (AI)</option>
          <option value="pvp">Two Players (Local)</option>
        </select>
      </div>

      <div class="control-group" id="difficulty-group">
        <label for="difficulty-select" class="control-label">AI Level</label>
        <select id="difficulty-select" class="custom-select">
          <option value="easy">Easy (Random)</option>
          <option value="medium">Medium (Tactical)</option>
          <option value="hard" selected>Hard (Minimax Unbeatable)</option>
        </select>
      </div>

      <div class="control-actions">
        <button id="sound-toggle-btn" class="icon-btn" title="Toggle Sound FX (Press M)">
          <span id="sound-icon">🔊</span>
        </button>
        <button id="help-btn" class="icon-btn" title="How to Play & Shortcuts">
          <span>❓</span>
        </button>
      </div>
    </section>

    <section class="status-bar">
      <div class="status-indicator">
        <span class="status-pulse" id="status-pulse"></span>
        <span id="status-text" class="status-text">Player X's Turn</span>
      </div>
      <div class="meta-stats">
        <span class="stat-pill" id="move-counter">Moves: 0</span>
        <span class="stat-pill" id="starter-pill">First: Player X</span>
      </div>
    </section>

    <main class="board-wrapper">
      <div id="game-board" class="game-board">
        <button class="cell" data-index="0" id="cell-0"><span class="mark-ghost"></span></button>
        <button class="cell" data-index="1" id="cell-1"><span class="mark-ghost"></span></button>
        <button class="cell" data-index="2" id="cell-2"><span class="mark-ghost"></span></button>
        <button class="cell" data-index="3" id="cell-3"><span class="mark-ghost"></span></button>
        <button class="cell" data-index="4" id="cell-4"><span class="mark-ghost"></span></button>
        <button class="cell" data-index="5" id="cell-5"><span class="mark-ghost"></span></button>
        <button class="cell" data-index="6" id="cell-6"><span class="mark-ghost"></span></button>
        <button class="cell" data-index="7" id="cell-7"><span class="mark-ghost"></span></button>
        <button class="cell" data-index="8" id="cell-8"><span class="mark-ghost"></span></button>

        <svg id="winning-strike" class="winning-strike" viewBox="0 0 300 300">
          <line id="strike-line" x1="0" y1="0" x2="0" y2="0" />
        </svg>
      </div>
    </main>

    <section class="scoreboard-card">
      <div class="score-item score-x">
        <div class="score-badge badge-x">X</div>
        <div class="score-info">
          <span class="score-label" id="label-x">Player X</span>
          <span class="score-value" id="score-x">0</span>
        </div>
      </div>
      <div class="score-divider"></div>
      <div class="score-item score-ties">
        <div class="score-badge badge-tie">🤝</div>
        <div class="score-info">
          <span class="score-label">Draws</span>
          <span class="score-value" id="score-ties">0</span>
        </div>
      </div>
      <div class="score-divider"></div>
      <div class="score-item score-o">
        <div class="score-badge badge-o">O</div>
        <div class="score-info">
          <span class="score-label" id="label-o">Computer (AI)</span>
          <span class="score-value" id="score-o">0</span>
        </div>
      </div>
    </section>

    <nav class="action-bar">
      <button id="undo-btn" class="action-btn secondary-btn" title="Undo Move (U)">
        <span>↩️ Undo</span>
        <kbd class="key-hint">U</kbd>
      </button>
      <button id="new-game-btn" class="action-btn primary-btn" title="New Game (N)">
        <span>✨ New Game</span>
        <kbd class="key-hint">N</kbd>
      </button>
      <button id="reset-scores-btn" class="action-btn secondary-btn" title="Reset Scores (R)">
        <span>🔄 Reset</span>
        <kbd class="key-hint">R</kbd>
      </button>
    </nav>

    <footer class="app-footer">
      <p class="shortcuts-legend">Tip: Press keys <kbd>1</kbd>–<kbd>9</kbd> on your keyboard / numpad to play a cell.</p>
    </footer>
  </div>

  <div id="result-modal" class="modal-overlay hidden">
    <div class="modal-card">
      <div class="modal-icon" id="modal-icon">🏆</div>
      <h2 id="modal-title" class="modal-title">Player X Wins!</h2>
      <p id="modal-subtitle" class="modal-subtitle">Great game! What would you like to do next?</p>
      <div class="modal-actions">
        <button id="modal-play-again-btn" class="action-btn primary-btn">Play Again</button>
        <button id="modal-close-btn" class="action-btn secondary-btn">Review Board</button>
      </div>
    </div>
  </div>

  <div id="help-modal" class="modal-overlay hidden">
    <div class="modal-card help-card">
      <div class="help-header">
        <h2 class="modal-title">Rules & Controls</h2>
        <button id="help-close-btn" class="close-btn">&times;</button>
      </div>
      <div class="help-content">
        <h3>Rules</h3>
        <p>Align 3 marks horizontally, vertically, or diagonally to win. If all 9 cells fill without a line, it is a Draw.</p>
        <h3>AI Difficulties</h3>
        <p><strong>Easy:</strong> Random moves.<br>
           <strong>Medium:</strong> Winning and defensive moves.<br>
           <strong>Hard:</strong> Unbeatable Minimax algorithm running in Python.</p>
        <h3>Keyboard Shortcuts</h3>
        <div class="shortcut-grid">
          <div><kbd>1</kbd> to <kbd>9</kbd></div><div>Play matching cell</div>
          <div><kbd>N</kbd></div><div>Start a new round</div>
          <div><kbd>U</kbd></div><div>Undo move</div>
          <div><kbd>R</kbd></div><div>Reset scores</div>
          <div><kbd>M</kbd></div><div>Toggle sound</div>
          <div><kbd>Esc</kbd></div><div>Close popups</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    (function () {
      const STRIKE_COORDINATES = {
        '0,1,2': { x1: 20, y1: 50, x2: 280, y2: 50 },
        '3,4,5': { x1: 20, y1: 150, x2: 280, y2: 150 },
        '6,7,8': { x1: 20, y1: 250, x2: 280, y2: 250 },
        '0,3,6': { x1: 50, y1: 20, x2: 50, y2: 280 },
        '1,4,7': { x1: 150, y1: 20, x2: 150, y2: 280 },
        '2,5,8': { x1: 250, y1: 20, x2: 250, y2: 280 },
        '0,4,8': { x1: 30, y1: 30, x2: 270, y2: 270 },
        '2,4,6': { x1: 270, y1: 30, x2: 30, y2: 270 }
      };

      class SoundFX {
        constructor() {
          this.ctx = null;
          this.enabled = localStorage.getItem('sound_enabled') !== 'false';
        }
        _init() {
          if (!this.ctx) {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) this.ctx = new AudioCtx();
          }
          if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume();
        }
        toggle() {
          this.enabled = !this.enabled;
          localStorage.setItem('sound_enabled', this.enabled);
          return this.enabled;
        }
        playMove(player) {
          if (!this.enabled) return;
          this._init();
          if (!this.ctx) return;
          const osc = this.ctx.createOscillator();
          const gain = this.ctx.createGain();
          const freq = player === 'X' ? 540 : 380;
          osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
          osc.frequency.exponentialRampToValueAtTime(freq * 1.3, this.ctx.currentTime + 0.08);
          gain.gain.setValueAtTime(0.12, this.ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + 0.1);
          osc.connect(gain); gain.connect(this.ctx.destination);
          osc.start(); osc.stop(this.ctx.currentTime + 0.1);
        }
        playWin() {
          if (!this.enabled) return;
          this._init();
          if (!this.ctx) return;
          [440, 554.37, 659.25, 880].forEach((freq, idx) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime + idx * 0.09);
            gain.gain.setValueAtTime(0.14, this.ctx.currentTime + idx * 0.09);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + idx * 0.09 + 0.35);
            osc.connect(gain); gain.connect(this.ctx.destination);
            osc.start(this.ctx.currentTime + idx * 0.09);
            osc.stop(this.ctx.currentTime + idx * 0.09 + 0.35);
          });
        }
        playDraw() {
          if (!this.enabled) return;
          this._init();
          if (!this.ctx) return;
          [380, 350, 320].forEach((freq, idx) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime + idx * 0.1);
            gain.gain.setValueAtTime(0.1, this.ctx.currentTime + idx * 0.1);
            gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + idx * 0.1 + 0.2);
            osc.connect(gain); gain.connect(this.ctx.destination);
            osc.start(this.ctx.currentTime + idx * 0.1);
            osc.stop(this.ctx.currentTime + idx * 0.1 + 0.2);
          });
        }
      }

      class ConfettiEngine {
        constructor(canvas) {
          this.canvas = canvas;
          this.ctx = canvas.getContext('2d');
          this.particles = [];
          this.anim = null;
          this.colors = ['#38bdf8', '#fb7185', '#34d399', '#fbbf24', '#ffffff'];
          this.resize();
          window.addEventListener('resize', () => this.resize());
        }
        resize() {
          this.canvas.width = window.innerWidth;
          this.canvas.height = window.innerHeight;
        }
        fire() {
          this.particles = [];
          for (let i = 0; i < 90; i++) {
            this.particles.push({
              x: this.canvas.width / 2, y: this.canvas.height / 2,
              w: Math.random() * 8 + 4, h: Math.random() * 8 + 4,
              color: this.colors[Math.floor(Math.random() * this.colors.length)],
              vx: (Math.random() - 0.5) * 18, vy: (Math.random() - 0.7) * 20,
              rot: Math.random() * 360, rotSpeed: (Math.random() - 0.5) * 12,
              gravity: 0.45, drag: 0.96, opacity: 1
            });
          }
          if (this.anim) cancelAnimationFrame(this.anim);
          this._render();
        }
        _render() {
          this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
          let alive = false;
          for (const p of this.particles) {
            p.vx *= p.drag; p.vy = p.vy * p.drag + p.gravity;
            p.x += p.vx; p.y += p.vy; p.rot += p.rotSpeed; p.opacity -= 0.012;
            if (p.opacity > 0) {
              alive = true;
              this.ctx.save();
              this.ctx.translate(p.x, p.y);
              this.ctx.rotate((p.rot * Math.PI) / 180);
              this.ctx.globalAlpha = Math.max(0, p.opacity);
              this.ctx.fillStyle = p.color;
              this.ctx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h);
              this.ctx.restore();
            }
          }
          if (alive) this.anim = requestAnimationFrame(() => this._render());
          else this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
      }

      class App {
        constructor() {
          this.sfx = new SoundFX();
          this.confetti = new ConfettiEngine(document.getElementById('confetti-canvas'));
          this.mode = 'pvc';
          this.difficulty = 'hard';
          this.isThinking = false;

          this.board = Array(9).fill(' ');
          this.currentPlayer = 'X';
          this.winner = null;
          this.winningLine = null;
          this.isGameOver = false;

          this._cacheDOM();
          this._bind();
          this._updateSoundIcon();
          this.fetchState();
        }

        _cacheDOM() {
          this.cells = Array.from(document.querySelectorAll('.cell'));
          this.statusText = document.getElementById('status-text');
          this.statusPulse = document.getElementById('status-pulse');
          this.moveCounter = document.getElementById('move-counter');
          this.starterPill = document.getElementById('starter-pill');
          this.modeSelect = document.getElementById('mode-select');
          this.diffGroup = document.getElementById('difficulty-group');
          this.diffSelect = document.getElementById('difficulty-select');
          this.scoreX = document.getElementById('score-x');
          this.scoreO = document.getElementById('score-o');
          this.scoreTies = document.getElementById('score-ties');
          this.labelO = document.getElementById('label-o');
          this.undoBtn = document.getElementById('undo-btn');
          this.newGameBtn = document.getElementById('new-game-btn');
          this.resetScoresBtn = document.getElementById('reset-scores-btn');
          this.soundBtn = document.getElementById('sound-toggle-btn');
          this.soundIcon = document.getElementById('sound-icon');
          this.helpBtn = document.getElementById('help-btn');
          this.helpCloseBtn = document.getElementById('help-close-btn');
          this.strikeSvg = document.getElementById('winning-strike');
          this.strikeLine = document.getElementById('strike-line');
          this.resultModal = document.getElementById('result-modal');
          this.helpModal = document.getElementById('help-modal');
          this.modalIcon = document.getElementById('modal-icon');
          this.modalTitle = document.getElementById('modal-title');
          this.modalSubtitle = document.getElementById('modal-subtitle');
          this.modalPlayAgain = document.getElementById('modal-play-again-btn');
          this.modalClose = document.getElementById('modal-close-btn');
        }

        _bind() {
          this.cells.forEach(cell => {
            const idx = parseInt(cell.getAttribute('data-index'), 10);
            cell.addEventListener('click', () => this.handleCellClick(idx));
            cell.addEventListener('mouseenter', () => this.handleCellHover(cell, idx));
            cell.addEventListener('mouseleave', () => this.handleCellLeave(cell));
          });

          this.modeSelect.addEventListener('change', (e) => {
            this.mode = e.target.value;
            if (this.mode === 'pvp') {
              this.diffGroup.style.display = 'none';
              this.labelO.textContent = 'Player O';
            } else {
              this.diffGroup.style.display = 'flex';
              this.labelO.textContent = 'Computer (AI)';
            }
            this.startNewGame();
          });

          this.diffSelect.addEventListener('change', (e) => {
            this.difficulty = e.target.value;
            this.startNewGame();
          });

          this.undoBtn.addEventListener('click', () => this.handleUndo());
          this.newGameBtn.addEventListener('click', () => this.startNewGame());
          this.resetScoresBtn.addEventListener('click', () => this.handleResetScores());

          this.soundBtn.addEventListener('click', () => {
            this.sfx.toggle();
            this._updateSoundIcon();
          });

          this.helpBtn.addEventListener('click', () => this.helpModal.classList.remove('hidden'));
          this.helpCloseBtn.addEventListener('click', () => this.helpModal.classList.add('hidden'));

          this.modalPlayAgain.addEventListener('click', () => {
            this.resultModal.classList.add('hidden');
            this.startNewGame();
          });
          this.modalClose.addEventListener('click', () => this.resultModal.classList.add('hidden'));

          window.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'SELECT') return;
            const key = e.key;
            if (key === 'Escape') {
              this.resultModal.classList.add('hidden');
              this.helpModal.classList.add('hidden');
            } else if (key >= '1' && key <= '9') {
              this.handleCellClick(parseInt(key, 10) - 1);
            } else if (key.toLowerCase() === 'n') {
              this.startNewGame();
            } else if (key.toLowerCase() === 'u') {
              this.handleUndo();
            } else if (key.toLowerCase() === 'r') {
              this.handleResetScores();
            } else if (key.toLowerCase() === 'm') {
              this.sfx.toggle();
              this._updateSoundIcon();
            }
          });
        }

        _updateSoundIcon() {
          this.soundIcon.textContent = this.sfx.enabled ? '🔊' : '🔇';
        }

        handleCellHover(cell, idx) {
          if (this.isGameOver || this.isThinking) return;
          if (this.board[idx] === ' ') {
            const ghost = cell.querySelector('.mark-ghost');
            if (ghost) {
              ghost.textContent = this.currentPlayer;
              ghost.style.color = this.currentPlayer === 'X' ? 'var(--color-x)' : 'var(--color-o)';
            }
          }
        }

        handleCellLeave(cell) {
          const ghost = cell.querySelector('.mark-ghost');
          if (ghost) ghost.textContent = '';
        }

        async fetchState() {
          try {
            const res = await fetch('/api/state');
            const data = await res.json();
            this.applyState(data);
          } catch (err) {
            console.error(err);
          }
        }

        async handleCellClick(index) {
          if (this.isThinking || this.isGameOver || this.board[index] !== ' ') return;

          // Optimistic local sound
          this.sfx.playMove(this.currentPlayer);

          // If playing against computer, show brief thinking indicator
          if (this.mode === 'pvc' && this.currentPlayer === 'X') {
            this.isThinking = true;
            this.statusText.textContent = 'Computer is thinking...';
            this.statusPulse.style.background = 'var(--color-o)';
            this.statusPulse.style.boxShadow = '0 0 10px var(--color-o)';
          }

          try {
            const res = await fetch('/api/move', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ index: index, mode: this.mode, difficulty: this.difficulty })
            });
            const data = await res.json();
            this.isThinking = false;
            this.applyState(data);

            if (data.game.isGameOver) {
              this._handleGameOver(data.game);
            }
          } catch (err) {
            this.isThinking = false;
            console.error(err);
          }
        }

        async handleUndo() {
          if (this.isThinking) return;
          try {
            const res = await fetch('/api/undo', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ mode: this.mode })
            });
            const data = await res.json();
            this.applyState(data);
          } catch (err) {
            console.error(err);
          }
        }

        async startNewGame() {
          this.isThinking = false;
          try {
            const res = await fetch('/api/new-game', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ mode: this.mode, difficulty: this.difficulty })
            });
            const data = await res.json();
            this.applyState(data);
            if (data.game.isGameOver) {
              this._handleGameOver(data.game);
            }
          } catch (err) {
            console.error(err);
          }
        }

        async handleResetScores() {
          if (confirm('Reset scores to 0?')) {
            try {
              const res = await fetch('/api/reset-scores', { method: 'POST' });
              const data = await res.json();
              this.applyState(data);
            } catch (err) {
              console.error(err);
            }
          }
        }

        applyState(data) {
          const game = data.game;
          this.board = game.board;
          this.currentPlayer = game.currentPlayer;
          this.winner = game.winner;
          this.winningLine = game.winningLine;
          this.isGameOver = game.isGameOver;

          this.moveCounter.textContent = 'Moves: ' + game.moveCount;
          this.starterPill.textContent = 'First: Player ' + game.startingPlayer;

          // Render cells
          this.cells.forEach((cell, idx) => {
            const val = this.board[idx];
            const existing = cell.querySelector('.mark-x, .mark-o');
            if (existing) existing.remove();
            cell.classList.remove('win-cell');

            if (val === 'X') {
              cell.classList.add('occupied');
              const span = document.createElement('span');
              span.className = 'mark-x'; span.textContent = 'X';
              cell.appendChild(span);
            } else if (val === 'O') {
              cell.classList.add('occupied');
              const span = document.createElement('span');
              span.className = 'mark-o'; span.textContent = 'O';
              cell.appendChild(span);
            } else {
              cell.classList.remove('occupied');
            }
          });

          // Status bar
          if (game.winner) {
            this.statusText.textContent = this.mode === 'pvc'
              ? (game.winner === 'X' ? 'You Won!' : 'Computer Won!')
              : `Player ${game.winner} Won!`;
            this.statusPulse.style.background = 'var(--color-win)';
            this.statusPulse.style.boxShadow = '0 0 10px var(--color-win)';
          } else if (game.isDraw) {
            this.statusText.textContent = "It's a Draw!";
            this.statusPulse.style.background = 'var(--text-muted)';
            this.statusPulse.style.boxShadow = 'none';
          } else {
            this.statusText.textContent = `Player ${game.currentPlayer}'s Turn`;
            const color = game.currentPlayer === 'X' ? 'var(--color-x)' : 'var(--color-o)';
            this.statusPulse.style.background = color;
            this.statusPulse.style.boxShadow = `0 0 10px ${color}`;
          }

          // Scores
          if (data.scores) {
            this.scoreX.textContent = data.scores.X;
            this.scoreO.textContent = data.scores.O;
            this.scoreTies.textContent = data.scores.ties;
          }

          // Strike line
          if (game.winningLine) {
            const key = game.winningLine.join(',');
            const coords = STRIKE_COORDINATES[key];
            if (coords) {
              this.strikeLine.setAttribute('x1', coords.x1);
              this.strikeLine.setAttribute('y1', coords.y1);
              this.strikeLine.setAttribute('x2', coords.x2);
              this.strikeLine.setAttribute('y2', coords.y2);
              this.strikeSvg.classList.add('active');
              requestAnimationFrame(() => this.strikeSvg.classList.add('draw'));
            }
            game.winningLine.forEach(idx => this.cells[idx].classList.add('win-cell'));
          } else {
            this.strikeSvg.classList.remove('active', 'draw');
          }
        }

        _handleGameOver(game) {
          if (game.winner) {
            if (this.mode === 'pvc' && game.winner === 'O') {
              this.sfx.playDraw();
            } else {
              this.sfx.playWin();
              this.confetti.fire();
            }
            const name = this.mode === 'pvc'
              ? (game.winner === 'X' ? 'You Win!' : 'Computer Wins!')
              : `Player ${game.winner} Wins!`;
            this.modalIcon.textContent = game.winner === 'X' ? '🏆' : (this.mode === 'pvc' ? '🤖' : '🎉');
            this.modalTitle.textContent = name;
            this.modalSubtitle.textContent = `Completed in ${game.moveCount} moves. Ready for another round?`;
          } else {
            this.sfx.playDraw();
            this.modalIcon.textContent = '🤝';
            this.modalTitle.textContent = "It's a Draw!";
            this.modalSubtitle.textContent = 'All 9 cells filled with no victor.';
          }
          setTimeout(() => this.resultModal.classList.remove('hidden'), 700);
        }
      }

      document.addEventListener('DOMContentLoaded', () => new App());
    })();
  </script>
</body>
</html>
"""


# ============================================================================
# HTTP SERVER & API CONTROLLER
# ============================================================================

class GameServerState:
    """Singleton server state containing active game and score manager."""

    def __init__(self) -> None:
        self.game: TicTacToeGame = TicTacToeGame(starting_player=PLAYER_X)
        self.score_manager: ScoreManager = ScoreManager(SCORES_FILE)
        self.next_starter: str = PLAYER_X
        self.lock: threading.Lock = threading.Lock()


# Global state instance shared across HTTP threads
SERVER_STATE = GameServerState()


class TicTacToeRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests and serves HTML UI and JSON REST endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP access logs for a clean console output."""
        return

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        """Helper to send JSON response."""
        response_bytes = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(response_bytes)

    def _send_html(self, html_content: str) -> None:
        """Helper to send HTML response."""
        response_bytes = html_content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def _read_json_body(self) -> Dict[str, Any]:
        """Read and parse JSON from the request payload."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:
        """Handle GET requests for page load or state inquiry."""
        url_path = urllib.parse.urlparse(self.path).path

        if url_path == "/" or url_path == "/index.html":
            self._send_html(HTML_PAGE)
        elif url_path == "/api/state":
            with SERVER_STATE.lock:
                self._send_json(200, {
                    "game": SERVER_STATE.game.to_dict(),
                    "scores": SERVER_STATE.score_manager.scores,
                })
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        """Handle POST requests for moves, resets, and undo operations."""
        url_path = urllib.parse.urlparse(self.path).path
        data = self._read_json_body()

        if url_path == "/api/move":
            cell_index = data.get("index")
            mode = data.get("mode", MODE_VS_COMPUTER)
            difficulty = data.get("difficulty", DIFFICULTY_HARD)

            if cell_index is None or not isinstance(cell_index, int):
                self._send_json(400, {"error": "Invalid cell index"})
                return

            with SERVER_STATE.lock:
                game = SERVER_STATE.game
                success = game.make_move(cell_index)

                if success and game.is_game_over():
                    SERVER_STATE.score_manager.record_result(game.winner)
                    SERVER_STATE.next_starter = (
                        PLAYER_O if game.starting_player == PLAYER_X else PLAYER_X
                    )
                elif success and mode == MODE_VS_COMPUTER and game.current_player == PLAYER_O:
                    # Compute AI move using Python Minimax engine
                    time.sleep(0.35)  # Natural thinking pause
                    ai_index = game.computer_move(difficulty)
                    if ai_index is not None:
                        game.make_move(ai_index)
                        if game.is_game_over():
                            SERVER_STATE.score_manager.record_result(game.winner)
                            SERVER_STATE.next_starter = (
                                PLAYER_O if game.starting_player == PLAYER_X else PLAYER_X
                            )

                self._send_json(200, {
                    "game": game.to_dict(),
                    "scores": SERVER_STATE.score_manager.scores,
                })

        elif url_path == "/api/new-game":
            mode = data.get("mode", MODE_VS_COMPUTER)
            difficulty = data.get("difficulty", DIFFICULTY_HARD)

            with SERVER_STATE.lock:
                # Alternate starting player each round
                starter = SERVER_STATE.next_starter
                SERVER_STATE.game.reset_board(starting_player=starter)

                # If Computer starts, execute AI move automatically
                if mode == MODE_VS_COMPUTER and SERVER_STATE.game.current_player == PLAYER_O:
                    time.sleep(0.35)
                    ai_index = SERVER_STATE.game.computer_move(difficulty)
                    if ai_index is not None:
                        SERVER_STATE.game.make_move(ai_index)

                self._send_json(200, {
                    "game": SERVER_STATE.game.to_dict(),
                    "scores": SERVER_STATE.score_manager.scores,
                })

        elif url_path == "/api/undo":
            mode = data.get("mode", MODE_VS_COMPUTER)

            with SERVER_STATE.lock:
                game = SERVER_STATE.game
                if mode == MODE_VS_COMPUTER:
                    # In single-player mode, revert both computer and player moves
                    if game.current_player == PLAYER_X and game.move_count >= 2:
                        game.undo_move()
                        game.undo_move()
                    else:
                        game.undo_move()
                else:
                    game.undo_move()

                self._send_json(200, {
                    "game": game.to_dict(),
                    "scores": SERVER_STATE.score_manager.scores,
                })

        elif url_path == "/api/reset-scores":
            with SERVER_STATE.lock:
                SERVER_STATE.score_manager.reset()
                self._send_json(200, {
                    "game": SERVER_STATE.game.to_dict(),
                    "scores": SERVER_STATE.score_manager.scores,
                })
        else:
            self.send_error(404, "Not Found")


# ============================================================================
# SERVER BOOTSTRAP & ENTRY POINT
# ============================================================================

def find_available_port(start_port: int = DEFAULT_PORT, max_attempts: int = 20) -> int:
    """Find a vacant TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((DEFAULT_HOST, port)) != 0:
                return port
    return start_port


def main() -> None:
    """Launch the Python HTTP Web Application and open the browser."""
    port = find_available_port(DEFAULT_PORT)
    server_address = (DEFAULT_HOST, port)
    app_url = f"http://{DEFAULT_HOST}:{port}"

    try:
        httpd = ThreadingHTTPServer(server_address, TicTacToeRequestHandler)
    except OSError as err:
        print(f"Error starting server on port {port}: {err}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("   🎮 TIC-TAC-TOE PYTHON WEB APPLICATION")
    print("=" * 60)
    print(f" • Server running at: {app_url}")
    print(" • Opening in your default web browser...")
    print(" • Press Ctrl+C in this terminal to stop the server.")
    print("=" * 60 + "\n")

    # Automatically open the browser after a brief 400ms delay
    def open_browser() -> None:
        time.sleep(0.4)
        webbrowser.open(app_url)

    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Tic-Tac-Toe server...")
        httpd.shutdown()
        httpd.server_close()
        print("Server stopped cleanly. Goodbye!\n")


if __name__ == "__main__":
    main()
