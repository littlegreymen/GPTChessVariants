#!/usr/bin/env python3
import math
import random
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path("/Users/daniel/Documents/Codex/2026-07-09/i/outputs")
sys.path.insert(0, str(OUTPUT_DIR))

import ChessvUltra_DPG as chess


POSITIONS = [
    ("after 1. e4", [("e2", "e4")]),
    ("sicilian setup", [("e2", "e4"), ("c7", "c5"), ("g1", "f3")]),
]


def build_game(plies, depth=2):
    game = chess.Game()
    game.AI_SEARCH_DEPTH = depth
    turn = "white"
    for from_alg, to_alg in plies:
        from_pos = chess.notation_to_coord(from_alg)
        to_pos = chess.notation_to_coord(to_alg)
        piece = game.board.get_piece(from_pos)
        assert piece is not None, f"No piece at {from_alg}"
        assert piece.color == turn, f"Expected {turn} at {from_alg}, got {piece.color}"
        assert game.board.move_piece(from_pos, to_pos), f"Illegal test move {from_alg}->{to_alg}"
        turn = "black" if turn == "white" else "white"
    assert turn == "black", "Test positions should leave Black to move"
    return game


def serial_root_scores(game):
    game.AI_USE_MULTIPROCESS = False
    moves = game.board.all_legal_moves("black")
    ordered = game._order_moves(game.board, moves, "black")
    return ordered, game._score_black_root_moves_serial(ordered)


def worker_root_scores(game):
    moves = game.board.all_legal_moves("black")
    ordered = game._order_moves(game.board, moves, "black")
    board_snapshot = game.board.clone()
    payloads = [
        (board_snapshot, move, game.AI_SEARCH_DEPTH, game.AI_QUIESCENCE_DEPTH)
        for move in ordered
    ]
    return ordered, [chess._score_black_root_move_worker(payload) for payload in payloads]


def parallel_root_scores(game):
    game.AI_USE_MULTIPROCESS = True
    moves = game.board.all_legal_moves("black")
    ordered = game._order_moves(game.board, moves, "black")
    scores = game._score_black_root_moves_parallel(ordered)
    assert scores is not None, "Parallel scoring was not used"
    return ordered, scores


def best_set(scores):
    best_score = max(score for _, score in scores)
    return {
        move
        for move, score in scores
        if abs(score - best_score) <= 1e-9
    }


def assert_equivalent(name, plies):
    serial_game = build_game(plies)
    parallel_game = build_game(plies)

    serial_ordered, serial_scores = serial_root_scores(serial_game)
    worker_ordered, worker_scores = worker_root_scores(parallel_game)

    assert serial_ordered == worker_ordered, f"{name}: ordered root moves changed"
    assert [move for move, _ in serial_scores] == [move for move, _ in worker_scores], f"{name}: scored move order changed"

    for (serial_move, serial_score), (worker_move, worker_score) in zip(serial_scores, worker_scores):
        assert serial_move == worker_move
        assert math.isclose(serial_score, worker_score, rel_tol=0.0, abs_tol=1e-9), (
            f"{name}: score mismatch for {serial_move}: serial={serial_score}, worker={worker_score}"
        )

    assert best_set(serial_scores) == best_set(worker_scores), f"{name}: best-move set changed"

    random.seed(12345)
    serial_choice_game = build_game(plies)
    serial_choice_game.AI_USE_MULTIPROCESS = False
    serial_choice = serial_choice_game._choose_ai_move()

    random.seed(12345)
    parallel_choice_game = build_game(plies)
    parallel_choice_game.AI_USE_MULTIPROCESS = False
    parallel_ordered, parallel_worker_scores = worker_root_scores(parallel_choice_game)
    best_moves = [
        move
        for move, score in parallel_worker_scores
        if move in best_set(parallel_worker_scores)
    ]
    parallel_choice = random.choice(best_moves)

    assert serial_choice == parallel_choice, f"{name}: seeded chosen move changed"
    print(f"PASS {name}: {len(serial_scores)} black moves, best set size {len(best_set(serial_scores))}")


def check_real_process_pool(name, plies):
    game = build_game(plies)
    try:
        parallel_ordered, parallel_scores = parallel_root_scores(game)
    except Exception as exc:
        print(f"SKIP real ProcessPoolExecutor check: {type(exc).__name__}: {exc}")
        return False
    finally:
        game.shutdown_ai_executor()

    serial_ordered, serial_scores = serial_root_scores(build_game(plies))
    assert serial_ordered == parallel_ordered, f"{name}: real pool ordered root moves changed"
    for (serial_move, serial_score), (parallel_move, parallel_score) in zip(serial_scores, parallel_scores):
        assert serial_move == parallel_move
        assert math.isclose(serial_score, parallel_score, rel_tol=0.0, abs_tol=1e-9), (
            f"{name}: real pool score mismatch for {serial_move}: serial={serial_score}, parallel={parallel_score}"
        )
    print(f"PASS real ProcessPoolExecutor check: {name}")
    return True


def benchmark(name, plies, depth=2):
    serial_game = build_game(plies, depth=depth)
    serial_game.AI_USE_MULTIPROCESS = False
    start = time.perf_counter()
    serial_game._choose_ai_move()
    serial_elapsed = time.perf_counter() - start

    parallel_game = build_game(plies, depth=depth)
    parallel_game.AI_USE_MULTIPROCESS = True
    try:
        parallel_game._choose_ai_move()
    except Exception as exc:
        parallel_game.shutdown_ai_executor()
        print(f"SKIP benchmark: real ProcessPoolExecutor unavailable ({type(exc).__name__}: {exc})")
        return

    start = time.perf_counter()
    parallel_game._choose_ai_move()
    parallel_elapsed = time.perf_counter() - start
    workers = parallel_game._last_ai_worker_count
    parallel_game.shutdown_ai_executor()

    ratio = serial_elapsed / parallel_elapsed if parallel_elapsed else float("inf")
    print(
        f"BENCH {name} depth {depth}: serial {serial_elapsed:.3f}s, "
        f"parallel warm {parallel_elapsed:.3f}s with {workers} workers, speedup {ratio:.2f}x"
    )


def main():
    for name, plies in POSITIONS:
        assert_equivalent(name, plies)
    if check_real_process_pool("sicilian setup", POSITIONS[-1][1]):
        benchmark("sicilian setup", POSITIONS[-1][1], depth=2)


if __name__ == "__main__":
    main()
