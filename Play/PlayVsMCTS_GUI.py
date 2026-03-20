import sys
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import subprocess
import time
import tkinter as tk

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Jeu.Yolah import Cell, Move, Square, Yolah
from Jeu.YolahInterface import YolahState
from MCTS.MCTS import mcts_collect_stats


class YolahGUI:
    BOARD_SIZE = 8
    CELL_SIZE = 72
    MARGIN = 34

    def __init__(self, root):
        self.root = root
        self.root.title("Yolah - Human vs MCTS")
        self.root.configure(bg="#f5efe2")

        self.state = YolahState()
        self.selected_from = None
        self.ai_thinking = False
        self.ai_queue = Queue()

        self.status_var = tk.StringVar(value="Black to move (you)")
        self.score_var = tk.StringVar(value="Score 0 - 0")

        top = tk.Frame(root, bg="#f5efe2")
        top.pack(padx=12, pady=(12, 6), fill="x")

        tk.Label(
            top,
            text="Yolah",
            font=("DejaVu Serif", 20, "bold"),
            fg="#3a2c1d",
            bg="#f5efe2",
        ).pack(anchor="w")

        tk.Label(
            top,
            textvariable=self.status_var,
            font=("DejaVu Sans", 12),
            fg="#4a3a28",
            bg="#f5efe2",
        ).pack(anchor="w")

        tk.Label(
            top,
            textvariable=self.score_var,
            font=("DejaVu Sans Mono", 11),
            fg="#4a3a28",
            bg="#f5efe2",
        ).pack(anchor="w")

        canvas_size = self.BOARD_SIZE * self.CELL_SIZE + 2 * self.MARGIN
        self.canvas = tk.Canvas(
            root,
            width=canvas_size,
            height=canvas_size,
            bg="#eadfc9",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=8)
        self.canvas.bind("<Button-1>", self.on_board_click)

        bottom = tk.Frame(root, bg="#f5efe2")
        bottom.pack(fill="x", padx=12, pady=(0, 12))

        tk.Button(
            bottom,
            text="New Game",
            command=self.new_game,
            bg="#5c8a5e",
            fg="white",
            activebackground="#476e49",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left")

        tk.Button(
            bottom,
            text="AI Move",
            command=self.start_ai_turn,
            bg="#7f5a3f",
            fg="white",
            activebackground="#664732",
            relief="flat",
            padx=12,
            pady=6,
        ).pack(side="left", padx=(8, 0))

        self.refresh_view()

    def new_game(self):
        self.state = YolahState()
        self.selected_from = None
        self.ai_thinking = False
        self.ai_queue = Queue()
        self.refresh_view()

    def square_to_screen(self, square):
        col = square.sq % self.BOARD_SIZE
        board_row = square.sq // self.BOARD_SIZE
        row = self.BOARD_SIZE - 1 - board_row
        x0 = self.MARGIN + col * self.CELL_SIZE
        y0 = self.MARGIN + row * self.CELL_SIZE
        return x0, y0

    def screen_to_square(self, x, y):
        bx = x - self.MARGIN
        by = y - self.MARGIN
        if bx < 0 or by < 0:
            return None

        col = bx // self.CELL_SIZE
        row = by // self.CELL_SIZE
        if col < 0 or col >= self.BOARD_SIZE or row < 0 or row >= self.BOARD_SIZE:
            return None

        board_row = self.BOARD_SIZE - 1 - int(row)
        idx = board_row * self.BOARD_SIZE + int(col)
        return Square(idx)

    def get_cell(self, square):
        bit = square.to_bitboard()
        game = self.state.game
        if game.black & bit:
            return Cell.BLACK
        if game.white & bit:
            return Cell.WHITE
        if game.empty & bit:
            return Cell.EMPTY
        return Cell.FREE

    def legal_moves(self):
        return self.state.legal_moves()

    def legal_from(self, from_square):
        return [m for m in self.legal_moves() if m != Move.none() and m.from_sq == from_square]

    def refresh_view(self):
        self.draw_board()
        b = self.state.game.black_score
        w = self.state.game.white_score
        self.score_var.set(f"Score {b} - {w}")

        if self.state.is_terminal():
            result = self.state.result()
            if result > 0:
                self.status_var.set("Game over - Black wins")
            elif result < 0:
                self.status_var.set("Game over - White wins")
            else:
                self.status_var.set("Game over - Draw")
            return

        if self.ai_thinking:
            self.status_var.set("White (AI) is thinking...")
        else:
            player = self.state.current_player()
            self.status_var.set("Black to move (you)" if player == Yolah.BLACK_PLAYER else "White to move (AI)")

    def draw_board(self):
        self.canvas.delete("all")

        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                x0 = self.MARGIN + col * self.CELL_SIZE
                y0 = self.MARGIN + row * self.CELL_SIZE
                x1 = x0 + self.CELL_SIZE
                y1 = y0 + self.CELL_SIZE
                color = "#d9b48f" if (row + col) % 2 == 0 else "#c79d73"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#9f7752")

        for i in range(self.BOARD_SIZE):
            file_txt = chr(ord("a") + i)
            rank_txt = str(self.BOARD_SIZE - i)
            x = self.MARGIN + i * self.CELL_SIZE + self.CELL_SIZE / 2
            y_top = self.MARGIN - 14
            y_bottom = self.MARGIN + self.BOARD_SIZE * self.CELL_SIZE + 14
            self.canvas.create_text(x, y_top, text=file_txt, fill="#503622", font=("DejaVu Sans", 10, "bold"))
            self.canvas.create_text(x, y_bottom, text=file_txt, fill="#503622", font=("DejaVu Sans", 10, "bold"))

            y = self.MARGIN + i * self.CELL_SIZE + self.CELL_SIZE / 2
            x_left = self.MARGIN - 14
            x_right = self.MARGIN + self.BOARD_SIZE * self.CELL_SIZE + 14
            self.canvas.create_text(x_left, y, text=rank_txt, fill="#503622", font=("DejaVu Sans", 10, "bold"))
            self.canvas.create_text(x_right, y, text=rank_txt, fill="#503622", font=("DejaVu Sans", 10, "bold"))

        possible_destinations = set()
        if self.selected_from is not None:
            for m in self.legal_from(self.selected_from):
                possible_destinations.add(m.to_sq.sq)

        for sq_idx in range(64):
            sq = Square(sq_idx)
            x0, y0 = self.square_to_screen(sq)
            x1 = x0 + self.CELL_SIZE
            y1 = y0 + self.CELL_SIZE

            if self.selected_from is not None and sq == self.selected_from:
                self.canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline="#f7f779", width=3)
            elif sq.sq in possible_destinations:
                self.canvas.create_oval(x0 + 28, y0 + 28, x1 - 28, y1 - 28, fill="#7cd95f", outline="")

            cell = self.get_cell(sq)
            if cell == Cell.BLACK:
                self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, fill="#1e1e1e", outline="#111", width=2)
            elif cell == Cell.WHITE:
                self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, fill="#f2f2f2", outline="#bcbcbc", width=2)
            elif cell == Cell.EMPTY:
                self.canvas.create_rectangle(x0 + 24, y0 + 24, x1 - 24, y1 - 24, fill="#6f5940", outline="")

    def on_board_click(self, event):
        if self.ai_thinking or self.state.is_terminal() or self.state.current_player() != Yolah.BLACK_PLAYER:
            return

        clicked = self.screen_to_square(event.x, event.y)
        if clicked is None:
            return

        moves = self.legal_moves()
        if moves == [Move.none()]:
            self.state.play(Move.none())
            self.selected_from = None
            self.refresh_view()
            self.start_ai_turn()
            return

        if self.selected_from is None:
            if self.legal_from(clicked):
                self.selected_from = clicked
                self.refresh_view()
            return

        if clicked == self.selected_from:
            self.selected_from = None
            self.refresh_view()
            return

        for m in self.legal_from(self.selected_from):
            if m.to_sq == clicked:
                self.state.play(m)
                self.selected_from = None
                self.refresh_view()
                self.start_ai_turn()
                return

        if self.legal_from(clicked):
            self.selected_from = clicked
        else:
            self.selected_from = None
        self.refresh_view()

    def start_ai_turn(self):
        if self.ai_thinking or self.state.is_terminal() or self.state.current_player() != Yolah.WHITE_PLAYER:
            return

        ai_moves = self.state.legal_moves()
        if ai_moves == [Move.none()]:
            self.state.play(Move.none())
            self.refresh_view()
            return

        self.ai_thinking = True
        self.refresh_view()

        snapshot = self.state.clone()
        thread = Thread(target=self._compute_ai_move, args=(snapshot,), daemon=True)
        thread.start()
        self.root.after(80, self._poll_ai_move)

    def _compute_ai_move(self, snapshot):
        # Prefer C++ engine for speed. Fallback to Python MCTS if binary is unavailable.
        t0 = time.perf_counter()
        game = snapshot.game
        engine_path = Path(__file__).resolve().with_name("PlayVsMCTS_fast")
        legal_moves = snapshot.legal_moves()
        legal_move_strs = {str(m) for m in legal_moves}

        if engine_path.exists():
            cmd = [
                str(engine_path),
                "--mcts",
                str(game.black),
                str(game.white),
                str(game.empty),
                str(game.black_score),
                str(game.white_score),
                str(game.ply),
                "120000",
                "2.5",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
                if result.returncode == 0:
                    move_str = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
                    if move_str in legal_move_strs:
                        elapsed = time.perf_counter() - t0
                        engine_stats = result.stderr.strip().splitlines()[-1].strip() if result.stderr.strip() else "iterations=unknown"
                        print(f"AI (C++) {engine_stats}, move={move_str}, total_elapsed={elapsed:.3f}s", flush=True)
                        self.ai_queue.put((Move.from_str(move_str), {"engine": "cpp"}))
                        return
                    print("AI (C++) returned an illegal/empty move, falling back to Python MCTS", flush=True)
                else:
                    err = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
                    print(f"AI (C++) failed with code {result.returncode}. {err}", flush=True)
            except Exception:
                print("AI (C++) execution failed, falling back to Python MCTS", flush=True)

        stats = mcts_collect_stats(snapshot, iterations=1600, time_limit_s=2.5)
        if not stats:
            elapsed = time.perf_counter() - t0
            print(f"AI (Python fallback) no stats, elapsed={elapsed:.3f}s", flush=True)
            self.ai_queue.put((Move.none(), stats))
            return

        best_move_str = max(stats.items(), key=lambda item: item[1][0])[0]
        elapsed = time.perf_counter() - t0
        print(f"AI (Python fallback) move={best_move_str}, elapsed={elapsed:.3f}s", flush=True)
        self.ai_queue.put((Move.from_str(best_move_str), stats))

    def _poll_ai_move(self):
        try:
            move, _stats = self.ai_queue.get_nowait()
        except Empty:
            self.root.after(80, self._poll_ai_move)
            return

        self.ai_thinking = False
        if not self.state.is_terminal() and self.state.current_player() == Yolah.WHITE_PLAYER:
            self.state.play(move)
        self.refresh_view()


def main():
    root = tk.Tk()
    app = YolahGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
