#!/usr/bin/env python3
"""
color_chess.py — single-file CLI chess with:
- pink/purple board (dark terminal)
- centered Unicode pieces
- last-move cyan highlight
- check/checkmate/stalemate detection
- pawn promotion to Queen
- en-passant support
- castling (king-side & queen-side)
- simple 1-ply material AI (black)
Input: e2->e4
"""
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

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

    def ai_move(self):
        moves = self.board.all_legal_moves("black")
        if not moves:
            return
        best_moves = []
        best_score = None
        for f,t in moves:
            sim = self.board.clone()
            sim.move_piece(f,t)
            score = sim.material_score()
            if best_score is None or score > best_score:
                best_score = score
                best_moves = [(f,t)]
            elif score == best_score:
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

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    random.seed()
    g = Game()
    try:
        g.play()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
        raise SystemExit(0)
