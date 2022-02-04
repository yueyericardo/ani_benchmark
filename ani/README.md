# Benchmark of Torchani

## Setup
```
conda create -n ani -c conda-forge torchani
conda activate ani
pip install ase
```

## Benchmark
```
python benchmark.py
```

## Result
Hardware:
- CPU: Intel(R) Core(TM) i7-6950X CPU @ 3.00GHz, with 10 Cores
- GPU: GeForce RTX 3080
```
device: cpu , energy       : 2.940 seconds
device: cpu , energy+force : 16.180 seconds
device: cuda, energy       : 0.711 seconds
device: cuda, energy+force : 0.831 seconds
```
