# Accelerated-Color-Lighting-Randomizer

## Overview
This package contains a C++/CUDA project that performs per-pixel randomized color augmentation (brightness, contrast, hue) using a batched GPU pipeline. Per-pixel randomization parameters are generated on-device using cuRAND to avoid host->device copies for parameters.

## Build (Linux)
Requirements:
- CUDA toolkit (11.x or newer recommended)
- OpenCV development libraries and headers
- CMake >= 3.18
- A CUDA-capable GPU

Build:
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Run
From `build/`:
```bash
./randomizer <input_image> <num_copies> <batch_size>
# Example:
./randomizer ../input.png 1000 8
```

This prints CPU and GPU timings and writes `randomized_output_gpu.png`.

## Contents
- `src/main.cu` — main program with device-side cuRAND generation of per-pixel params, streams, pinned memory.
- `src/utils.h` — helper macros.
- `tools/py_cuda/randomizer.py` — PyCUDA prototype.
- `bench/run_bench.py` — simple runner to save timing output.

