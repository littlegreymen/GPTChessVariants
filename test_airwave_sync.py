#!/usr/bin/env python3
import sys
from pathlib import Path

OUTPUT_DIR = Path("/Users/daniel/Documents/Codex/2026-07-09/i/outputs")
sys.path.insert(0, str(OUTPUT_DIR))

import ChessvUltra_DPG_Airwave as chess


def setup_sound_game(color, game_id="ABC123"):
    game = chess.DPGChessGame()
    game.game_mode = game.MODE_SOUND
    game.local_color = color
    game.game_id = game_id
    game.new_game(send_reset=False)
    return game


def make_local_sound_message(game, from_alg, to_alg):
    from_pos = chess.notation_to_coord(from_alg)
    to_pos = chess.notation_to_coord(to_alg)
    assert game._make_gui_move(from_pos, to_pos, validate=True)
    game.turn = game._opponent_color(game.local_color)
    game.sound_move_no += 1
    move_text = from_alg + to_alg
    return game._encode_airwave_message(
        "MOVE",
        game.game_id,
        game.sound_move_no,
        move_text,
        game._board_sync_hash(),
    )


def main():
    white = setup_sound_game("white")
    black = setup_sound_game("black")

    white_move = make_local_sound_message(white, "e2", "e4")
    black._handle_airwave_text(white_move)
    assert black._board_sync_hash() == white._board_sync_hash()
    assert black.turn == "black"
    assert black.sound_move_no == 1

    black_move = make_local_sound_message(black, "c7", "c5")
    white._handle_airwave_text(black_move)
    assert white._board_sync_hash() == black._board_sync_hash()
    assert white.turn == "white"
    assert white.sound_move_no == 2

    before_hash = white._board_sync_hash()
    bad_move = white._encode_airwave_message("MOVE", white.game_id, 3, "g1f3", "badcafe0")
    white._handle_airwave_text(bad_move)
    assert white._board_sync_hash() == before_hash

    joiner = setup_sound_game("black", game_id=None)
    joiner._handle_airwave_text("CVU1|HELLO|ZZ99|white")
    assert joiner.game_id == "ZZ99"

    print("PASS airwave sync message tests")


if __name__ == "__main__":
    main()
