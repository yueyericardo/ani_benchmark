# Benchmark of TorchANI for 20k ethanol

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


Ref: [Permutationally invariant polynomial regression for energies and gradients, using reverse differentiation, achieves orders of magnitude speed-up with high precision compared to other machine learning methods: The Journal of Chemical Physics: Vol 156, No 4](https://aip.scitation.org/doi/suppl/10.1063/5.0080506)
