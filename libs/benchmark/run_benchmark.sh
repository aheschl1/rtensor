#!/bin/bash
# Benchmark runner script that sets up the Python environment

export LD_LIBRARY_PATH="/home/$USER/miniconda3/lib:$LD_LIBRARY_PATH"
cargo run "$@" --release --features cuda
