#!/usr/bin/env python3
import sys
from pathlib import Path


OUTPUT_DIR = Path("/Users/daniel/Documents/Codex/2026-07-09/i/outputs")
sys.path.insert(0, str(OUTPUT_DIR))

import ChessvUltra_DPG_nn as chess


def test_fen_generation():
    game = chess.Game()
    assert game.board_to_fen("white") == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert game.board.move_piece(chess.notation_to_coord("e2"), chess.notation_to_coord("e4"))
    assert game.board_to_fen("black") == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_fake_nn_bestmove():
    game = chess.DPGChessGame()
    game.black_player_mode = game.BLACK_MODE_NN
    game.nn_engine_command = f"{sys.executable} {OUTPUT_DIR / 'fake_uci_engine.py'}"
    game.nn_movetime_ms = 10

    assert game.board.move_piece(chess.notation_to_coord("e2"), chess.notation_to_coord("e4"))
    game.turn = "black"
    move = game._choose_nn_move()
    game.shutdown_nn_engine()

    assert move == (chess.notation_to_coord("g8"), chess.notation_to_coord("f6"))


def main():
    test_fen_generation()
    test_fake_nn_bestmove()
    print("PASS NN engine integration tests")


if __name__ == "__main__":
    main()
