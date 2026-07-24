#!/usr/bin/env python3
import sys


def send(line):
    print(line, flush=True)


def main():
    for raw in sys.stdin:
        line = raw.strip()
        if line == "uci":
            send("id name Fake NN Test Engine")
            send("uciok")
        elif line == "isready":
            send("readyok")
        elif line.startswith("go "):
            send("bestmove g8f6")
        elif line == "quit":
            return


if __name__ == "__main__":
    main()
