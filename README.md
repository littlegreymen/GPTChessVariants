# GPTChessVariants

This repository contains several standalone chess programs:

- Three terminal games: Basic, Super, and Ultra.
- DearPyGui applications, including a timing version and a multiprocessing version.
- An Airwave version that can send moves using sound.
- A neural-engine version that can connect to an external UCI-compatible engine.

## Requirements

Use 64-bit Python 3.11 for the smoothest package compatibility. The Python packages used by the GUI and optional sound features are listed in requirements.txt.

### Create the virtual environment

Run these commands from the repository folder.

macOS or Linux:

    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Windows PowerShell:

    py -3.11 -m venv .venv
    ./.venv/Scripts/Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

If you use the Airwave sound mode, install the PortAudio system library before installing the Python packages:

- macOS with Homebrew: brew install portaudio
- Debian or Ubuntu: sudo apt install build-essential portaudio19-dev python3-dev
- Windows: the PyAudio package from pip includes PortAudio.

The Airwave mode also needs a working microphone and speakers. The other chess versions do not need audio hardware.

## Run a game

Run any command below from the repository folder with the virtual environment activated.

Terminal games:

    python ChessvBasic.py
    python ChessvSuper.py
    python ChessvUltra.py

DearPyGui games:

    python ChessvUltra_DPG.py
    python ChessvUltra_DPG_multiprocess.py
    python chess_ultraDPG_timing.py

Sound transport:

    python ChessvUltra_DPG_Airwave.py

Neural-engine version:

    python ChessvUltra_DPG_nn.py

The neural-engine version can use its built-in AI. To use neural mode, install lc0 separately with its network weights, then make the lc0 command available on PATH. To use a different executable or command, set the CHESSVULTRA_NN_ENGINE environment variable before starting the script.

## Run the included checks

These standalone checks use only Python's standard library plus the project files:

    python test_parallel_ai.py
    python test_nn_engine_integration.py
    python test_airwave_sync.py
