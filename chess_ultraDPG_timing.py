#!/usr/bin/env python3
"""
ChessvUltra_DPG.py — single-file DearPyGui chess with:
- pink/purple board (dark terminal)
- centered Unicode pieces
- last-move cyan highlight
- check/checkmate/stalemate detection
- pawn promotion to Queen
- en-passant support
- castling (king-side & queen-side)
- improved built-in AI (black)
- click a source square, then a target square
"""
import random
import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

try:
    import dearpygui.dearpygui as dpg
except ModuleNotFoundError:
    dpg = None

# --------------------------
# Terminal colors
# --------------------------
RESET = "\033[0m"
BG_LIGHT = "\033[105m"  # bright magenta (pink)
BG_DARK = "\033[45m"    # magenta-ish (purple)
FG_WHITE = "\033[97m"
FG_BLACK = "\033[30m"
CYAN = "\033[46m"       # cyan background for last move
BOLD = "\033[1m"

# --------------------------
# Helpers
# --------------------------
def notation_to_coord(s: str) -> Tuple[int, int]:
    s = s.strip()
    file = s[0].lower()
    rank = int(s[1])
    col = ord(file) - ord("a")
    row = 8 - rank
    return row, col

def coord_to_notation(pos: Tuple[int,int]) -> str:
    row, col = pos
    return f"{chr(col + ord('a'))}{8 - row}"

# --------------------------
# Piece base and concrete pieces
# --------------------------
class ChessPiece(ABC):
    def __init__(self, color: str, position: Tuple[int,int]):
        self.color = color
        self.position = position
        self.has_moved = False

    @abstractmethod
    def symbol(self) -> str: pass

    @abstractmethod
    def pseudo_moves(self, board) -> List[Tuple[int,int]]: pass

    def move(self, new_pos: Tuple[int,int]):
        self.position = new_pos
        self.has_moved = True

    def clone(self):
        new = type(self)(self.color, self.position)
        new.has_moved = getattr(self, "has_moved", False)
        return new

class Pawn(ChessPiece):
    def symbol(self): return "♙" if self.color=="white" else "♟"
    def pseudo_moves(self, board):
        row,col = self.position
        moves=[]
        direction = -1 if self.color=="white" else 1
        # forward one
        one = (row + direction, col)
        if board.is_on_board(one) and board.is_empty(one):
            moves.append(one)
            # forward two from start
            start_row = 6 if self.color=="white" else 1
            two = (row + 2*direction, col)
            if row == start_row and board.is_on_board(two) and board.is_empty(two):
                moves.append(two)
        # captures (including en-passant)
        for dc in (-1, 1):
            diag = (row + direction, col + dc)
            if board.is_on_board(diag):
                if board.is_enemy(diag, self.color):
                    moves.append(diag)
                # en-passant: diag equals en_passant_target (even though empty)
                if board.en_passant_target and diag == board.en_passant_target:
                    moves.append(diag)
        return moves

class Rook(ChessPiece):
    def symbol(self): return "♖" if self.color=="white" else "♜"
    def pseudo_moves(self, board):
        return board._sliding_moves(self, [(1,0),(-1,0),(0,1),(0,-1)])

class Bishop(ChessPiece):
    def symbol(self): return "♗" if self.color=="white" else "♝"
    def pseudo_moves(self, board):
        return board._sliding_moves(self, [(1,1),(1,-1),(-1,1),(-1,-1)])

class Queen(ChessPiece):
    def symbol(self): return "♕" if self.color=="white" else "♛"
    def pseudo_moves(self, board):
        return board._sliding_moves(self, [
            (1,0),(-1,0),(0,1),(0,-1),
            (1,1),(1,-1),(-1,1),(-1,-1)
        ])

class Knight(ChessPiece):
    def symbol(self): return "♘" if self.color=="white" else "♞"
    def pseudo_moves(self, board):
        row,col = self.position
        offsets=[(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
        out=[]
        for dr,dc in offsets:
            p=(row+dr,col+dc)
            if board.is_on_board(p) and not board.is_friendly(p,self.color):
                out.append(p)
        return out

class King(ChessPiece):
    def symbol(self): return "♔" if self.color=="white" else "♚"
    def pseudo_moves(self, board):
        row,col = self.position
        out=[]
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0: continue
                p=(row+dr,col+dc)
                if board.is_on_board(p) and not board.is_friendly(p,self.color):
                    out.append(p)

        # CASTLING: king-side and queen-side — rely on board helpers
        if not self.has_moved and not board.is_in_check(self.color):
            # king-side
            if board.can_castle_kingside(self.color):
                # king goes two squares toward rook
                out.append((row, col+2))
            # queen-side
            if board.can_castle_queenside(self.color):
                out.append((row, col-2))
        return out

# --------------------------
# ChessBoard with castling & en-passant support
# --------------------------
class ChessBoard:
    def __init__(self):
        self.board: Dict[Tuple[int,int], ChessPiece] = {}
        self.setup_board()
        # En-passant target (square that can be captured onto) and captured pawn position
        self.en_passant_target: Optional[Tuple[int,int]] = None
        self.en_passant_capture_pos: Optional[Tuple[int,int]] = None

    def setup_board(self):
        self.board={}
        for c in range(8):
            self.board[(6,c)] = Pawn("white",(6,c))
            self.board[(1,c)] = Pawn("black",(1,c))
        # rooks
        self.board[(7,0)] = Rook("white",(7,0))
        self.board[(7,7)] = Rook("white",(7,7))
        self.board[(0,0)] = Rook("black",(0,0))
        self.board[(0,7)] = Rook("black",(0,7))
        # knights
        self.board[(7,1)] = Knight("white",(7,1))
        self.board[(7,6)] = Knight("white",(7,6))
        self.board[(0,1)] = Knight("black",(0,1))
        self.board[(0,6)] = Knight("black",(0,6))
        # bishops
        self.board[(7,2)] = Bishop("white",(7,2))
        self.board[(7,5)] = Bishop("white",(7,5))
        self.board[(0,2)] = Bishop("black",(0,2))
        self.board[(0,5)] = Bishop("black",(0,5))
        # queens
        self.board[(7,3)] = Queen("white",(7,3))
        self.board[(0,3)] = Queen("black",(0,3))
        # kings
        self.board[(7,4)] = King("white",(7,4))
        self.board[(0,4)] = King("black",(0,4))
        # reset en-passant
        self.en_passant_target = None
        self.en_passant_capture_pos = None

    # --------------------------
    # Basic helpers
    # --------------------------
    def is_on_board(self,pos): r,c = pos; return 0<=r<8 and 0<=c<8
    def get_piece(self,pos): return self.board.get(pos)
    def is_empty(self,pos): return self.get_piece(pos) is None
    def is_friendly(self,pos,color):
        p=self.get_piece(pos); return (p is not None) and p.color==color
    def is_enemy(self,pos,color):
        p=self.get_piece(pos); return (p is not None) and p.color!=color

    def _sliding_moves(self,piece,directions):
        row,col = piece.position
        moves=[]
        for dr,dc in directions:
            r,c = row+dr, col+dc
            while self.is_on_board((r,c)):
                if self.is_empty((r,c)):
                    moves.append((r,c))
                else:
                    if self.is_enemy((r,c), piece.color):
                        moves.append((r,c))
                    break
                r += dr; c += dc
        return moves

    def clone(self):
        nb = ChessBoard.__new__(ChessBoard)
        nb.board = {}
        for pos,piece in self.board.items():
            # clone each piece and keep has_moved
            new_piece = piece.clone()
            nb.board[pos] = new_piece
        nb.en_passant_target = self.en_passant_target
        nb.en_passant_capture_pos = self.en_passant_capture_pos
        return nb

    def find_king(self,color) -> Optional[Tuple[int,int]]:
        for pos,p in self.board.items():
            if isinstance(p, King) and p.color==color:
                return pos
        return None

    # --------------------------
    # Attack detection
    # --------------------------
    def is_square_attacked(self, pos, by_color):
        """Return True if the square is attacked by any piece of by_color."""
        for row in range(8):
            for col in range(8):
                piece = self.get_piece((row, col))
                if piece and piece.color == by_color:
                    # ⚠️ Avoid infinite recursion:
                    if isinstance(piece, King):
                        # The king attacks only one square in each direction
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                if dr == 0 and dc == 0:
                                    continue
                                if (row + dr, col + dc) == pos:
                                    return True
                    else:
                        if pos in piece.pseudo_moves(self):
                            return True
        return False
                  
    def _attacks(self,piece,target):
        # pawn attacks are diagonal
        if isinstance(piece, Pawn):
            pr,pc = piece.position
            dir = -1 if piece.color == "white" else 1
            for dc in (-1,1):
                if (pr + dir, pc + dc) == target:
                    return True
            return False
        # knights, king, sliding and others via pseudo_moves but note that pseudo_moves for pawns uses en_passant_target so we avoid that
        if isinstance(piece, Knight) or isinstance(piece, King):
            return target in piece.pseudo_moves(self)
        if isinstance(piece, Rook):
            dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        elif isinstance(piece, Bishop):
            dirs = [(1,1),(1,-1),(-1,1),(-1,-1)]
        elif isinstance(piece, Queen):
            dirs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        else:
            # fallback
            return target in piece.pseudo_moves(self)

        pr,pc = piece.position
        for dr,dc in dirs:
            r,c = pr + dr, pc + dc
            while self.is_on_board((r,c)):
                if (r,c) == target:
                    return True
                if not self.is_empty((r,c)):
                    break
                r += dr; c += dc
        return False

    # --------------------------
    # Castling checks
    # --------------------------
    def can_castle_kingside(self, color: str) -> bool:
        king_pos = self.find_king(color)
        if king_pos is None: return False
        king = self.get_piece(king_pos)
        if king.has_moved: return False
        row, col = king_pos
        rook_pos = (row, 7)
        rook = self.get_piece(rook_pos)
        if not rook or rook.has_moved or not isinstance(rook, Rook): return False
        # squares between king and rook must be empty: cols 5 and 6
        for c in (5,6):
            if not self.is_empty((row,c)):
                return False
        # king cannot be in check, and squares king passes through (5 and 6) must not be attacked
        if self.is_in_check(color): return False
        for c in (5,6):
            if self.is_square_attacked((row,c), "white" if color=="black" else "black"):
                return False
        return True

    def can_castle_queenside(self, color: str) -> bool:
        king_pos = self.find_king(color)
        if king_pos is None: return False
        king = self.get_piece(king_pos)
        if king.has_moved: return False
        row, col = king_pos
        rook_pos = (row, 0)
        rook = self.get_piece(rook_pos)
        if not rook or rook.has_moved or not isinstance(rook, Rook): return False
        # squares between king and rook must be empty: cols 1,2,3
        for c in (1,2,3):
            if not self.is_empty((row,c)):
                return False
        # king cannot be in check, and squares king passes through (3 and 2) must not be attacked
        if self.is_in_check(color): return False
        for c in (3,2):
            if self.is_square_attacked((row,c), "white" if color=="black" else "black"):
                return False
        return True

    # --------------------------
    # Check detection & legal moves
    # --------------------------
    def is_in_check(self, color: str) -> bool:
        king_pos = self.find_king(color)
        if not king_pos: return False
        return self.is_square_attacked(king_pos, "white" if color=="black" else "black")

    def legal_moves_for(self,pos: Tuple[int,int]) -> List[Tuple[int,int]]:
        piece = self.get_piece(pos)
        if piece is None: return []
        pseudo = piece.pseudo_moves(self)
        legal = []
        for to in pseudo:
            sim = self.clone()
            mp = sim.get_piece(pos)
            if mp is None:
                continue
            # Execute move on simulation — handle captures, castling, en-passant, promotions similarly to move_piece
            # Remove captured normally
            if to in sim.board:
                del sim.board[to]
            # detect castling on sim: king moving two columns -> also move rook
            from_pos = pos
            fr,fc = from_pos; tr,tc = to
            # delete source
            del sim.board[from_pos]
            mp.move(to)
            sim.board[to] = mp
            # handle en-passant capture if applicable
            if isinstance(mp, Pawn):
                # if moved to en_passant_target, remove the captured pawn at en_passant_capture_pos
                if sim.en_passant_target and to == sim.en_passant_target and sim.en_passant_capture_pos:
                    if sim.en_passant_capture_pos in sim.board:
                        del sim.board[sim.en_passant_capture_pos]
                # promotion in simulation
                r,_ = to
                if (mp.color=="white" and r==0) or (mp.color=="black" and r==7):
                    sim.board[to] = Queen(mp.color, to)
            # handle castling rook move on simulation
            if isinstance(mp, King) and abs(tc - fc) == 2:
                # king-side
                if tc > fc:
                    rook_from = (fr, 7)
                    rook_to = (fr, 5)
                else:
                    rook_from = (fr, 0)
                    rook_to = (fr, 3)
                rook_piece = sim.board.get(rook_from)
                if rook_piece:
                    del sim.board[rook_from]
                    rook_piece.move(rook_to)
                    sim.board[rook_to] = rook_piece
            # After sim move, clear en_passant_target unless this sim move creates a new double pawn move (we didn't set that above)
            sim.en_passant_target = None
            sim.en_passant_capture_pos = None

            # If move leaves own king in check, it's illegal
            if not sim.is_in_check(piece.color):
                legal.append(to)
        return legal

    def all_legal_moves(self,color: str) -> List[Tuple[Tuple[int,int],Tuple[int,int]]]:
        out=[]
        for pos,piece in self.board.items():
            if piece.color != color: continue
            for to in self.legal_moves_for(pos):
                out.append((pos,to))
        return out

    # --------------------------
    # Material eval (AI)
    # --------------------------
    def material_score(self) -> int:
        values = {Pawn:1, Knight:3, Bishop:3, Rook:5, Queen:9, King:0}
        score = 0
        for p in self.board.values():
            for cls,val in values.items():
                if isinstance(p, cls):
                    score += val if p.color=="black" else -val
                    break
        return score

    # --------------------------
    # Make a move (handles en-passant & castling & promotion)
    # --------------------------
    def move_piece(self, from_pos: Tuple[int,int], to_pos: Tuple[int,int]) -> bool:
        piece = self.get_piece(from_pos)
        if piece is None:
            print("No piece at that position.")
            return False

        legal = self.legal_moves_for(from_pos)
        if to_pos not in legal:
            print("Illegal move.")
            return False

        fr,fc = from_pos
        tr,tc = to_pos

        # Handle captures (normal capture)
        captured = self.get_piece(to_pos)
        # Handle en-passant capture if moving pawn to en_passant_target
        if isinstance(piece, Pawn) and self.en_passant_target and to_pos == self.en_passant_target and self.en_passant_capture_pos:
            # remove the pawn that moved two squares previously
            if self.en_passant_capture_pos in self.board:
                captured = self.board[self.en_passant_capture_pos]
                del self.board[self.en_passant_capture_pos]
                print(f"{piece.color.capitalize()} pawn captures en-passant!")

        if captured and captured != self.get_piece(to_pos):
            # captured already removed in en-passant; if not, print capture (we will print below)
            pass

        if captured and to_pos in self.board:
            print(f"{piece.color.capitalize()} {type(piece).__name__} captures {captured.color} {type(captured).__name__}!")

        # Remove piece from source
        del self.board[from_pos]

        # Castling: move rook as well if king moves two squares
        if isinstance(piece, King) and abs(tc - fc) == 2:
            # Determine rook positions and move rook
            if tc > fc:
                # king-side
                rook_from = (fr, 7)
                rook_to = (fr, 5)
            else:
                rook_from = (fr, 0)
                rook_to = (fr, 3)
            rook = self.get_piece(rook_from)
            if rook:
                del self.board[rook_from]
                rook.move(rook_to)
                self.board[rook_to] = rook

        # Normal move
        piece.move(to_pos)

        # Pawn promotion
        if isinstance(piece, Pawn):
            r,_ = to_pos
            if (piece.color == "white" and r == 0) or (piece.color == "black" and r == 7):
                piece = Queen(piece.color, to_pos)
                print(f"{piece.color.capitalize()} pawn promoted to Queen!")

        # Place moved piece
        # (If target had piece, it was overwritten earlier by deletion)
        self.board[to_pos] = piece

        # --- handle en-passant target setting/reset ---
        # reset defaults first
        self.en_passant_target = None
        self.en_passant_capture_pos = None
        # if a pawn moved two squares, set en-passant target to the passed square, and capture pos to the destination pawn pos
        if isinstance(piece, Pawn) and abs(tr - fr) == 2:
            passed_row = (tr + fr) // 2
            passed_col = fc
            self.en_passant_target = (passed_row, passed_col)
            self.en_passant_capture_pos = (tr, tc)  # pawn that can be captured (the pawn's current square)
        return True

    # --------------------------
    # Display with last-move highlight
    # --------------------------
    def display(self, cell_w: int = 6, cell_h: int = 3,
                last_move: Optional[Tuple[Tuple[int,int],Tuple[int,int]]] = None):
        for row in range(8):
            for subline in range(cell_h):
                row_str = ""
                left_rank = f"{8 - row}" if subline == cell_h // 2 else " "
                row_str += left_rank + " "
                for col in range(8):
                    piece = self.get_piece((row,col))
                    fg = FG_WHITE if piece and piece.color == "white" else FG_BLACK
                    if last_move and ((row,col) == last_move[0] or (row,col) == last_move[1]):
                        bg = CYAN
                    else:
                        bg = BG_LIGHT if (row + col) % 2 == 0 else BG_DARK
                    if subline == cell_h // 2:
                        padding = (cell_w - 1) // 2
                        symbol = piece.symbol() if piece else " "
                        content = " " * padding + symbol + " " * (cell_w - 1 - padding)
                    else:
                        content = " " * cell_w
                    row_str += f"{bg}{fg}{content}{RESET}"
                right_rank = f" {8 - row}" if subline == cell_h // 2 else "  "
                row_str += right_rank
                print(row_str)
        bottom_label = " " * (cell_w // 2 + 2)
        bottom_files = "".join(f" {chr(ord('a') + c)} ".center(cell_w) for c in range(8))
        print(bottom_label + bottom_files)
        print()

# --------------------------
# Game controller
# --------------------------
class Game:
    AI_SEARCH_DEPTH = 2
    AI_QUIESCENCE_DEPTH = 1
    CHECKMATE_SCORE = 100000.0
    PIECE_VALUES = {
        Pawn: 100,
        Knight: 320,
        Bishop: 330,
        Rook: 500,
        Queen: 900,
        King: 0,
    }
    PIECE_SQUARE_TABLES = {
        Pawn: (
            0, 0, 0, 0, 0, 0, 0, 0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5, 5, 10, 25, 25, 10, 5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, -5, -10, 0, 0, -10, -5, 5,
            5, 10, 10, -20, -20, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0,
        ),
        Knight: (
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50,
        ),
        Bishop: (
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10, 5, 0, 0, 0, 0, 5, -10,
            -10, 10, 10, 10, 10, 10, 10, -10,
            -10, 0, 10, 10, 10, 10, 0, -10,
            -10, 5, 5, 10, 10, 5, 5, -10,
            -10, 0, 5, 10, 10, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20,
        ),
        Rook: (
            0, 0, 0, 5, 5, 0, 0, 0,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0,
        ),
        Queen: (
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10, 0, 5, 0, 0, 0, 0, -10,
            -10, 5, 5, 5, 5, 5, 0, -10,
            0, 0, 5, 5, 5, 5, 0, -5,
            -5, 0, 5, 5, 5, 5, 0, -5,
            -10, 0, 5, 5, 5, 5, 0, -10,
            -10, 0, 0, 0, 0, 0, 0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20,
        ),
        King: (
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20,
        ),
    }

    def __init__(self):
        self.board = ChessBoard()
        self.turn = "white"
        self.last_move: Optional[Tuple[Tuple[int,int],Tuple[int,int]]] = None

    def human_move(self):
        while True:
            move = input(f"{self.turn.capitalize()} move (e.g. e2->e4 or 'quit'): ").strip()
            if not move:
                continue
            if move.lower() in ("quit","exit"):
                print("Goodbye.")
                raise SystemExit(0)
            if "->" not in move:
                print("Use format e2->e4")
                continue
            left,right = move.split("->")
            try:
                from_pos = notation_to_coord(left.strip())
                to_pos = notation_to_coord(right.strip())
            except Exception:
                print("Invalid notation. Use e2->e4")
                continue
            piece = self.board.get_piece(from_pos)
            if not piece:
                print("No piece at source.")
                continue
            if piece.color != self.turn:
                print("That's not your piece.")
                continue
            if self.board.move_piece(from_pos,to_pos):
                self.last_move = (from_pos,to_pos)
                break

    def _piece_square_bonus(self, piece, pos) -> int:
        table = self.PIECE_SQUARE_TABLES.get(type(piece))
        if table is None:
            return 0
        row, col = pos
        table_row = row if piece.color == "white" else 7 - row
        return table[table_row * 8 + col]

    def _material_position_score(self, board_sim) -> float:
        score = 0.0
        bishops = {"black": 0, "white": 0}
        pawn_files = {"black": [0] * 8, "white": [0] * 8}

        for pos, piece in board_sim.board.items():
            if isinstance(piece, Pawn):
                pawn_files[piece.color][pos[1]] += 1

        for pos, piece in board_sim.board.items():
            sign = 1 if piece.color == "black" else -1
            score += sign * (
                self.PIECE_VALUES.get(type(piece), 0) +
                self._piece_square_bonus(piece, pos)
            )
            if isinstance(piece, Bishop):
                bishops[piece.color] += 1

            home_row = 0 if piece.color == "black" else 7
            if isinstance(piece, (Knight, Bishop)) and pos[0] == home_row and not piece.has_moved:
                score -= sign * 12
            if isinstance(piece, Rook):
                friendly_pawns = pawn_files[piece.color][pos[1]]
                enemy_color = "white" if piece.color == "black" else "black"
                enemy_pawns = pawn_files[enemy_color][pos[1]]
                if friendly_pawns == 0:
                    score += sign * (12 if enemy_pawns == 0 else 6)

        if bishops["black"] >= 2:
            score += 30
        if bishops["white"] >= 2:
            score -= 30
        return score

    def _pawn_structure_score(self, board_sim) -> float:
        pawns = {"black": [], "white": []}
        for pos, piece in board_sim.board.items():
            if isinstance(piece, Pawn):
                pawns[piece.color].append(pos)

        score = 0.0
        for color in ("black", "white"):
            sign = 1 if color == "black" else -1
            files = [0] * 8
            for _, col in pawns[color]:
                files[col] += 1

            enemy = "white" if color == "black" else "black"
            enemy_pawns = pawns[enemy]
            for row, col in pawns[color]:
                if files[col] > 1:
                    score -= sign * 10
                if (col == 0 or files[col - 1] == 0) and (col == 7 or files[col + 1] == 0):
                    score -= sign * 12

                is_passed = True
                for erow, ecol in enemy_pawns:
                    if abs(ecol - col) <= 1:
                        if color == "black" and erow > row:
                            is_passed = False
                            break
                        if color == "white" and erow < row:
                            is_passed = False
                            break
                if is_passed:
                    advancement = row if color == "black" else 7 - row
                    passed_bonus = (0, 6, 12, 22, 36, 58, 90, 0)[advancement]
                    score += sign * passed_bonus
        return score

    def _king_safety_score(self, board_sim) -> float:
        score = 0.0
        for color in ("black", "white"):
            sign = 1 if color == "black" else -1
            king_pos = board_sim.find_king(color)
            if king_pos is None:
                score -= sign * self.CHECKMATE_SCORE
                continue

            row, col = king_pos
            if (color == "black" and king_pos in ((0, 6), (0, 2))) or (color == "white" and king_pos in ((7, 6), (7, 2))):
                score += sign * 35
            elif col in (3, 4) and not board_sim.get_piece(king_pos).has_moved:
                score -= sign * 12

            shield_row = row + (1 if color == "black" else -1)
            if 0 <= shield_row < 8:
                for dc in (-1, 0, 1):
                    shield_piece = board_sim.get_piece((shield_row, col + dc))
                    if isinstance(shield_piece, Pawn) and shield_piece.color == color:
                        score += sign * 8

        if board_sim.is_in_check("black"):
            score -= 55
        if board_sim.is_in_check("white"):
            score += 55
        return score

    def _mobility_score(self, board_sim) -> float:
        try:
            black_moves = 0
            white_moves = 0
            for piece in board_sim.board.values():
                if piece.color == "black":
                    black_moves += len(piece.pseudo_moves(board_sim))
                else:
                    white_moves += len(piece.pseudo_moves(board_sim))
            return 1.5 * (black_moves - white_moves)
        except Exception:
            return 0.0

    def evaluate_board(self, board_sim) -> float:
        if board_sim.find_king("white") is None:
            return self.CHECKMATE_SCORE
        if board_sim.find_king("black") is None:
            return -self.CHECKMATE_SCORE
        return (
            self._material_position_score(board_sim) +
            self._pawn_structure_score(board_sim) +
            self._king_safety_score(board_sim) +
            self._mobility_score(board_sim)
        )

    def _is_capture_or_promotion(self, board_sim, move) -> bool:
        f, t = move
        piece = board_sim.get_piece(f)
        if piece is None:
            return False
        if board_sim.get_piece(t) is not None:
            return True
        if isinstance(piece, Pawn):
            if board_sim.en_passant_target and t == board_sim.en_passant_target:
                return True
            return (piece.color == "black" and t[0] == 7) or (piece.color == "white" and t[0] == 0)
        return False

    def _static_move_score(self, board_sim, move) -> float:
        f, t = move
        piece = board_sim.get_piece(f)
        if piece is None:
            return 0.0

        target = board_sim.get_piece(t)
        score = 0.0
        if target:
            victim = self.PIECE_VALUES.get(type(target), 0)
            attacker = self.PIECE_VALUES.get(type(piece), 0)
            score += 10000 + 10 * victim - attacker
        elif isinstance(piece, Pawn) and board_sim.en_passant_target and t == board_sim.en_passant_target:
            score += 10000 + 10 * self.PIECE_VALUES[Pawn] - self.PIECE_VALUES[Pawn]

        if isinstance(piece, Pawn) and ((piece.color == "black" and t[0] == 7) or (piece.color == "white" and t[0] == 0)):
            score += 9000
        if isinstance(piece, King) and abs(t[1] - f[1]) == 2:
            score += 700
        if isinstance(piece, (Knight, Bishop)) and f[0] in (0, 7):
            score += 120
        if t in ((3, 3), (3, 4), (4, 3), (4, 4)):
            score += 40
        return score

    def _order_moves(self, board_sim, moves, for_color):
        scored = [(self._static_move_score(board_sim, move), move) for move in moves]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored]

    def _board_signature(self, board_sim):
        pieces = []
        for (row, col), piece in sorted(board_sim.board.items()):
            pieces.append((row, col, type(piece).__name__, piece.color, piece.has_moved))
        return (tuple(pieces), board_sim.en_passant_target, board_sim.en_passant_capture_pos)

    def _apply_move_for_search(self, board_sim, from_pos, to_pos):
        piece = board_sim.get_piece(from_pos)
        if piece is None:
            return False

        fr, fc = from_pos
        tr, tc = to_pos
        if isinstance(piece, Pawn) and board_sim.en_passant_target and to_pos == board_sim.en_passant_target and board_sim.en_passant_capture_pos:
            board_sim.board.pop(board_sim.en_passant_capture_pos, None)

        board_sim.board.pop(to_pos, None)
        del board_sim.board[from_pos]

        if isinstance(piece, King) and abs(tc - fc) == 2:
            rook_from = (fr, 7) if tc > fc else (fr, 0)
            rook_to = (fr, 5) if tc > fc else (fr, 3)
            rook = board_sim.get_piece(rook_from)
            if rook:
                del board_sim.board[rook_from]
                rook.move(rook_to)
                board_sim.board[rook_to] = rook

        piece.move(to_pos)
        if isinstance(piece, Pawn) and ((piece.color == "white" and tr == 0) or (piece.color == "black" and tr == 7)):
            piece = Queen(piece.color, to_pos)
        board_sim.board[to_pos] = piece

        board_sim.en_passant_target = None
        board_sim.en_passant_capture_pos = None
        if isinstance(piece, Pawn) and abs(tr - fr) == 2:
            board_sim.en_passant_target = ((tr + fr) // 2, fc)
            board_sim.en_passant_capture_pos = (tr, tc)
        return True

    def _quiescence(self, board_sim, alpha: float, beta: float, maximizing: bool, depth: int) -> float:
        stand_pat = self.evaluate_board(board_sim)
        if depth == 0:
            return stand_pat

        if maximizing:
            value = stand_pat
            if value >= beta:
                return value
            if value > alpha:
                alpha = value
            color = "black"
            moves = [move for move in board_sim.all_legal_moves(color) if self._is_capture_or_promotion(board_sim, move)]
            for f, t in self._order_moves(board_sim, moves, color):
                sim = board_sim.clone()
                self._apply_move_for_search(sim, f, t)
                score = self._quiescence(sim, alpha, beta, False, depth - 1)
                if score > value:
                    value = score
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
            return value

        value = stand_pat
        if value <= alpha:
            return value
        if value < beta:
            beta = value
        color = "white"
        moves = [move for move in board_sim.all_legal_moves(color) if self._is_capture_or_promotion(board_sim, move)]
        for f, t in self._order_moves(board_sim, moves, color):
            sim = board_sim.clone()
            self._apply_move_for_search(sim, f, t)
            score = self._quiescence(sim, alpha, beta, True, depth - 1)
            if score < value:
                value = score
            if value < beta:
                beta = value
            if alpha >= beta:
                break
        return value

    def _minimax(self, board_sim, depth:int, alpha:float, beta:float, maximizing:bool, ply:int=0) -> float:
        if depth == 0:
            return self._quiescence(board_sim, alpha, beta, maximizing, self.AI_QUIESCENCE_DEPTH)

        color = "black" if maximizing else "white"
        cache_key = (depth, maximizing, alpha, beta, self._board_signature(board_sim))
        if cache_key in self._search_cache:
            return self._search_cache[cache_key]

        moves = board_sim.all_legal_moves(color)
        if not moves:
            if board_sim.is_in_check(color):
                return (-self.CHECKMATE_SCORE + ply) if maximizing else (self.CHECKMATE_SCORE - ply)
            return 0.0

        ordered = self._order_moves(board_sim, moves, color)
        cutoff = False
        if maximizing:
            value = -self.CHECKMATE_SCORE
            for f,t in ordered:
                sim = board_sim.clone()
                self._apply_move_for_search(sim, f, t)
                val = self._minimax(sim, depth-1, alpha, beta, False, ply+1)
                if val > value: value = val
                if value > alpha: alpha = value
                if alpha >= beta:
                    cutoff = True
                    break
        else:
            value = self.CHECKMATE_SCORE
            for f,t in ordered:
                sim = board_sim.clone()
                self._apply_move_for_search(sim, f, t)
                val = self._minimax(sim, depth-1, alpha, beta, True, ply+1)
                if val < value: value = val
                if value < beta: beta = value
                if alpha >= beta:
                    cutoff = True
                    break

        if not cutoff:
            self._search_cache[cache_key] = value
        return value

    def ai_move(self):
        moves = self.board.all_legal_moves("black")
        if not moves:
            return

        self._search_cache = {}
        ordered = self._order_moves(self.board, moves, "black")
        best_score = -self.CHECKMATE_SCORE
        best_moves = []
        for f,t in ordered:
            sim = self.board.clone()
            self._apply_move_for_search(sim, f, t)
            val = self._minimax(
                sim,
                depth=self.AI_SEARCH_DEPTH,
                alpha=-self.CHECKMATE_SCORE,
                beta=self.CHECKMATE_SCORE,
                maximizing=False,
                ply=1,
            )
            if val > best_score + 1e-9:
                best_score = val
                best_moves = [(f,t)]
            elif abs(val - best_score) <= 1e-9:
                best_moves.append((f,t))
        chosen = random.choice(best_moves)
        f,t = chosen
        self.board.move_piece(f,t)
        self.last_move = (f,t)
        print(f"Black moves {coord_to_notation(f)}->{coord_to_notation(t)}")

    def play(self):
        print("Welcome — you are White. AI is Black.")
        print("Enter moves like: e2->e4. Type 'quit' to exit.")
        print()
        while True:
            self.board.display(last_move=self.last_move)
            if self.board.is_in_check(self.turn):
                print(BOLD + f"{self.turn.capitalize()} is in check!" + RESET)
            moves = self.board.all_legal_moves(self.turn)
            if not moves:
                if self.board.is_in_check(self.turn):
                    winner = "Black" if self.turn == "white" else "White"
                    print(BOLD + f"Checkmate! {winner} wins!" + RESET)
                else:
                    print(BOLD + "Stalemate!" + RESET)
                break
            if self.turn == "white":
                self.human_move()
                self.turn = "black"
            else:
                self.ai_move()
                self.turn = "white"

class DPGChessGame(Game):
    BOARD_SQUARE_SIZE = 62
    SIDE_PANEL_WIDTH = 520
    LOG_HEIGHT = 440
    LIGHT_BG = (218, 74, 196, 255)
    DARK_BG = (136, 51, 132, 255)
    LAST_BG = (0, 188, 212, 255)
    SELECTED_BG = (245, 203, 92, 255)
    LEGAL_BG = (91, 186, 146, 255)
    WHITE_FG = (255, 255, 255, 255)
    BLACK_FG = (12, 12, 16, 255)
    EMPTY_FG = (34, 24, 38, 255)
    GUI_FALLBACK_SYMBOLS = {
        ("white", Pawn): "P",
        ("white", Knight): "N",
        ("white", Bishop): "B",
        ("white", Rook): "R",
        ("white", Queen): "Q",
        ("white", King): "K",
        ("black", Pawn): "p",
        ("black", Knight): "n",
        ("black", Bishop): "b",
        ("black", Rook): "r",
        ("black", Queen): "q",
        ("black", King): "k",
    }
    PIECE_FONT_CANDIDATES = (
        "/System/Library/Fonts/Apple Symbols.ttf",
        "/System/Library/Fonts/Symbol.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    )

    def __init__(self):
        super().__init__()
        self.selected_square = None
        self.legal_targets = set()
        self.game_over = False
        self.last_ai_time_seconds = None
        self.log_lines = []
        self.window_tag = "chessvultra_dpg_window"
        self.status_tag = "chessvultra_dpg_status"
        self.ai_time_tag = "chessvultra_dpg_ai_time"
        self.log_tag = "chessvultra_dpg_log"
        self.piece_font_tag = "chessvultra_piece_font"
        self.piece_font_loaded = False
        self.undo_stack = []

    def _square_tag(self, row, col):
        return f"chessvultra_square_{row}_{col}"

    def _theme_tag(self, bg_name, fg_name):
        return f"chessvultra_theme_{bg_name}_{fg_name}"

    def _brighten(self, color, amount):
        r, g, b, a = color
        return (
            max(0, min(255, r + amount)),
            max(0, min(255, g + amount)),
            max(0, min(255, b + amount)),
            a,
        )

    def _load_piece_font(self):
        font_path = next((path for path in self.PIECE_FONT_CANDIDATES if os.path.exists(path)), None)
        if font_path is None:
            return

        try:
            with dpg.font_registry():
                with dpg.font(font_path, 36, tag=self.piece_font_tag):
                    dpg.add_font_range(0x2654, 0x2660)
            self.piece_font_loaded = True
        except Exception:
            self.piece_font_loaded = False

    def _gui_piece_symbol(self, piece):
        if piece is None:
            return " "
        if self.piece_font_loaded:
            return piece.symbol()
        for cls in (Pawn, Knight, Bishop, Rook, Queen, King):
            if isinstance(piece, cls):
                return self.GUI_FALLBACK_SYMBOLS[(piece.color, cls)]
        return "?"

    def _create_square_themes(self):
        backgrounds = {
            "light": self.LIGHT_BG,
            "dark": self.DARK_BG,
            "last": self.LAST_BG,
            "selected": self.SELECTED_BG,
            "legal": self.LEGAL_BG,
        }
        foregrounds = {
            "white": self.WHITE_FG,
            "black": self.BLACK_FG,
            "empty": self.EMPTY_FG,
        }
        for bg_name, bg in backgrounds.items():
            for fg_name, fg in foregrounds.items():
                with dpg.theme(tag=self._theme_tag(bg_name, fg_name)):
                    with dpg.theme_component(dpg.mvButton):
                        dpg.add_theme_color(dpg.mvThemeCol_Button, bg)
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, self._brighten(bg, 22))
                        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, self._brighten(bg, -24))
                        dpg.add_theme_color(dpg.mvThemeCol_Text, fg)
                        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0)

    def _set_status(self, message):
        if dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, message)

    def _format_ai_time(self, seconds):
        if seconds is None:
            return "Black AI time: --"
        if seconds < 1:
            return f"Black AI time: {seconds * 1000:.0f} ms"
        return f"Black AI time: {seconds:.2f} s"

    def _sync_ai_time(self):
        if dpg.does_item_exist(self.ai_time_tag):
            dpg.set_value(self.ai_time_tag, self._format_ai_time(self.last_ai_time_seconds))

    def _log(self, message):
        self.log_lines.append(message)
        self.log_lines = self.log_lines[-80:]
        self._sync_log()

    def _sync_log(self):
        if dpg.does_item_exist(self.log_tag):
            dpg.set_value(self.log_tag, "\n".join(self.log_lines))

    def _reset_selection(self):
        self.selected_square = None
        self.legal_targets = set()

    def _snapshot_state(self):
        return {
            "board": self.board.clone(),
            "turn": self.turn,
            "last_move": self.last_move,
            "game_over": self.game_over,
            "last_ai_time_seconds": self.last_ai_time_seconds,
            "log_lines": list(self.log_lines),
        }

    def _restore_state(self, state):
        self.board = state["board"].clone()
        self.turn = state["turn"]
        self.last_move = state["last_move"]
        self.game_over = state["game_over"]
        self.last_ai_time_seconds = state.get("last_ai_time_seconds")
        self.log_lines = list(state["log_lines"])
        self._reset_selection()
        self._sync_log()
        self._sync_ai_time()

    def _piece_fg_name(self, piece):
        if piece is None:
            return "empty"
        return "white" if piece.color == "white" else "black"

    def refresh_board(self):
        for row in range(8):
            for col in range(8):
                pos = (row, col)
                tag = self._square_tag(row, col)
                if not dpg.does_item_exist(tag):
                    continue

                piece = self.board.get_piece(pos)
                dpg.configure_item(tag, label=self._gui_piece_symbol(piece))

                if pos == self.selected_square:
                    bg_name = "selected"
                elif pos in self.legal_targets:
                    bg_name = "legal"
                elif self.last_move and pos in self.last_move:
                    bg_name = "last"
                else:
                    bg_name = "light" if (row + col) % 2 == 0 else "dark"

                dpg.bind_item_theme(tag, self._theme_tag(bg_name, self._piece_fg_name(piece)))

    def _update_status_for_turn(self):
        moves = self.board.all_legal_moves(self.turn)
        if not moves:
            self.game_over = True
            self._reset_selection()
            if self.board.is_in_check(self.turn):
                winner = "Black" if self.turn == "white" else "White"
                self._set_status(f"Checkmate! {winner} wins!")
                self._log(f"Checkmate! {winner} wins!")
            else:
                self._set_status("Stalemate!")
                self._log("Stalemate!")
            return True

        if self.board.is_in_check(self.turn):
            self._set_status(f"{self.turn.capitalize()} is in check!")
        elif self.turn == "white":
            self._set_status("White to move. Click a source square, then a target square.")
        else:
            self._set_status("Black to move.")
        return False

    def _move_messages(self, from_pos, to_pos):
        piece = self.board.get_piece(from_pos)
        if piece is None:
            return []

        messages = []
        target = self.board.get_piece(to_pos)
        if isinstance(piece, Pawn) and self.board.en_passant_target and to_pos == self.board.en_passant_target and self.board.en_passant_capture_pos:
            messages.append(f"{piece.color.capitalize()} pawn captures en-passant!")
        elif target:
            messages.append(f"{piece.color.capitalize()} {type(piece).__name__} captures {target.color} {type(target).__name__}!")

        messages.append(f"{piece.color.capitalize()} moves {coord_to_notation(from_pos)}->{coord_to_notation(to_pos)}")
        if isinstance(piece, Pawn) and ((piece.color == "white" and to_pos[0] == 0) or (piece.color == "black" and to_pos[0] == 7)):
            messages.append(f"{piece.color.capitalize()} pawn promoted to Queen!")
        return messages

    def _make_gui_move(self, from_pos, to_pos, validate=True):
        if validate and to_pos not in self.board.legal_moves_for(from_pos):
            self._set_status("Illegal move.")
            return False

        messages = self._move_messages(from_pos, to_pos)
        if not self._apply_move_for_search(self.board, from_pos, to_pos):
            self._set_status("Illegal move.")
            return False

        self.last_move = (from_pos, to_pos)
        for message in messages:
            self._log(message)
        return True

    def _choose_ai_move(self):
        moves = self.board.all_legal_moves("black")
        if not moves:
            return None

        self._search_cache = {}
        ordered = self._order_moves(self.board, moves, "black")
        best_score = -self.CHECKMATE_SCORE
        best_moves = []
        for f, t in ordered:
            sim = self.board.clone()
            self._apply_move_for_search(sim, f, t)
            val = self._minimax(
                sim,
                depth=self.AI_SEARCH_DEPTH,
                alpha=-self.CHECKMATE_SCORE,
                beta=self.CHECKMATE_SCORE,
                maximizing=False,
                ply=1,
            )
            if val > best_score + 1e-9:
                best_score = val
                best_moves = [(f, t)]
            elif abs(val - best_score) <= 1e-9:
                best_moves.append((f, t))
        return random.choice(best_moves)

    def _make_ai_reply(self):
        self._set_status("Black is thinking...")
        start_time = time.perf_counter()
        move = self._choose_ai_move()
        elapsed = time.perf_counter() - start_time
        if move is None:
            self._update_status_for_turn()
            self.refresh_board()
            return

        self._make_gui_move(move[0], move[1], validate=False)
        self.last_ai_time_seconds = elapsed
        self._sync_ai_time()
        self._log(f"Black AI took {elapsed:.3f} seconds.")
        self.turn = "white"
        self._reset_selection()
        self._update_status_for_turn()
        self.refresh_board()

    def on_square_click(self, sender, app_data, user_data):
        if self.game_over:
            return
        if self.turn != "white":
            self._set_status("Black is thinking.")
            return

        pos = user_data
        piece = self.board.get_piece(pos)
        if self.selected_square is None:
            if piece is None:
                self._set_status("Select a white piece first.")
                return
            if piece.color != "white":
                self._set_status("That's not your piece.")
                return
            self.selected_square = pos
            self.legal_targets = set(self.board.legal_moves_for(pos))
            self._set_status(f"Selected {coord_to_notation(pos)}. Click a target square.")
            self.refresh_board()
            return

        if pos == self.selected_square:
            self._reset_selection()
            self._update_status_for_turn()
            self.refresh_board()
            return

        if piece is not None and piece.color == "white" and pos not in self.legal_targets:
            self.selected_square = pos
            self.legal_targets = set(self.board.legal_moves_for(pos))
            self._set_status(f"Selected {coord_to_notation(pos)}. Click a target square.")
            self.refresh_board()
            return

        from_pos = self.selected_square
        if pos not in self.legal_targets:
            self._set_status("Illegal move.")
            return

        undo_state = self._snapshot_state()
        if self._make_gui_move(from_pos, pos):
            self.undo_stack.append(undo_state)
            self.turn = "black"
            self._reset_selection()
            self.refresh_board()
            if not self._update_status_for_turn():
                self._make_ai_reply()

    def undo_white_move(self, sender=None, app_data=None, user_data=None):
        if not self.undo_stack:
            self._set_status("No white move to undo.")
            return

        self._restore_state(self.undo_stack.pop())
        self._update_status_for_turn()
        if self.turn == "white":
            if self.board.is_in_check("white"):
                self._set_status("Undid White's last move. White is in check.")
            else:
                self._set_status("Undid White's last move. White to move.")
        else:
            self._set_status("Undid White's last move.")
        self.refresh_board()

    def new_game(self, sender=None, app_data=None, user_data=None):
        self.board = ChessBoard()
        self.turn = "white"
        self.last_move = None
        self.game_over = False
        self.last_ai_time_seconds = None
        self._reset_selection()
        self.undo_stack = []
        self.log_lines = []
        self._log("Welcome - you are White. AI is Black.")
        self._log("Click a source square, then a target square.")
        self._sync_ai_time()
        self._update_status_for_turn()
        self.refresh_board()

    def run(self):
        dpg.create_context()
        self._load_piece_font()
        self._create_square_themes()

        with dpg.window(label="ChessvUltra DPG", tag=self.window_tag):
            dpg.add_text("", tag=self.status_tag)
            dpg.add_text(self._format_ai_time(self.last_ai_time_seconds), tag=self.ai_time_tag)
            dpg.add_spacer(height=8)
            with dpg.group(horizontal=True):
                with dpg.group():
                    for row in range(8):
                        with dpg.group(horizontal=True):
                            for col in range(8):
                                dpg.add_button(
                                    label=" ",
                                    tag=self._square_tag(row, col),
                                    width=self.BOARD_SQUARE_SIZE,
                                    height=self.BOARD_SQUARE_SIZE,
                                    callback=self.on_square_click,
                                    user_data=(row, col),
                                )
                                if self.piece_font_loaded:
                                    dpg.bind_item_font(self._square_tag(row, col), self.piece_font_tag)
                with dpg.group():
                    dpg.add_button(label="New Game", width=self.SIDE_PANEL_WIDTH, callback=self.new_game)
                    dpg.add_button(label="Undo White Move", width=self.SIDE_PANEL_WIDTH, callback=self.undo_white_move)
                    dpg.add_button(label="Quit", width=self.SIDE_PANEL_WIDTH, callback=lambda: dpg.stop_dearpygui())
                    dpg.add_spacer(height=8)
                    dpg.add_text("Move log")
                    dpg.add_input_text(
                        tag=self.log_tag,
                        multiline=True,
                        readonly=True,
                        width=self.SIDE_PANEL_WIDTH,
                        height=self.LOG_HEIGHT,
                    )

        self.new_game()
        dpg.create_viewport(title="ChessvUltra:DearPyGui:@Littlegreymen", width=1070, height=590)
        dpg.setup_dearpygui()
        dpg.set_primary_window(self.window_tag, True)
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    random.seed()
    if dpg is None:
        print("DearPyGui is required to run this version.")
        print("Install it with: python3 -m pip install dearpygui")
        raise SystemExit(1)
    DPGChessGame().run()
