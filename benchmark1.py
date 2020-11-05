import mdtraj
import time
import torch
import torchani
import pynvml
import gc
import os
import subprocess
import argparse
from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions


def checkgpu(device=None):
    i = device if device else torch.cuda.current_device()
    real_i = int(os.environ['CUDA_VISIBLE_DEVICES'][0]) if 'CUDA_VISIBLE_DEVICES' in os.environ else i
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    name = pynvml.nvmlDeviceGetName(h)
    print('  GPU Memory Used (nvidia-smi): {:7.1f}MB / {:.1f}MB ({})'.format(info.used / 1024 / 1024, info.total / 1024 / 1024, name.decode()))


def alert(text):
    print('\033[91m{}\33[0m'.format(text))  # red


def info(text):
    print('\033[32m{}\33[0m'.format(text))  # green


def benchmark(speciesPositions, aev_comp, N, check_gpu_mem):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    start = time.time()

    for i in range(N):
        aev = aev_comp(speciesPositions).aevs
        if i == 2 and check_gpu_mem:
            checkgpu()

    torch.cuda.synchronize()
    delta = time.time() - start
    print(f'  Duration: {delta:.2f} s')
    print(f'  Speed: {delta/N*1000:.2f} ms/it')
    return aev, delta


def check_speedup_error(aev, aev_ref, speed, speed_ref):
    speedUP = speed_ref / speed
    if speedUP > 1:
        info(f'  Speed up: {speedUP:.2f} X\n')
    else:
        alert(f'  Speed up (slower): {speedUP:.2f} X\n')

    aev_error = torch.max(torch.abs(aev - aev_ref))
    assert aev_error < 0.02, f'  Error: {aev_error:.1e}\n'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--nnpops_cu_out',
                        default=None)
    parser.add_argument('-c', '--check_gpu_mem',
                        dest='check_gpu_mem',
                        action='store_const',
                        const=1)
    parser.set_defaults(check_gpu_mem=0)
    parser = parser.parse_args()

    check_gpu_mem = parser.check_gpu_mem
    device = torch.device('cuda')
    # files = ['2iuz_ligand.mol2', '1hvk_ligand.mol2', '2iuz_ligand.mol2', '3hkw_ligand.mol2', '3hky_ligand.mol2', '3lka_ligand.mol2', '3o99_ligand.mol2', 'small.pdb', '1hz5.pdb', '6W8H.pdb']
    files = ['small.pdb', '1hz5.pdb', '6W8H.pdb']

    for file in files:
        mol = mdtraj.load(f'molecules/{file}')
        species = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
        positions = torch.tensor(mol.xyz * 10, dtype=torch.float32, requires_grad=False, device=device)
        print(f'File: {file}, Molecule size: {species.shape[-1]}\n')

        nnp = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
        speciesPositions = nnp.species_converter((species, positions))
        symmFuncRef = nnp.aev_computer
        symmFunc = TorchANISymmetryFunctions(nnp.aev_computer).to(device)

        N = 100

        print('Original TorchANI:')
        aev_ref, delta_ref = benchmark(speciesPositions, symmFuncRef, N, check_gpu_mem)
        print()

        print('NNPops:')
        aev, delta = benchmark(speciesPositions, symmFunc, N, check_gpu_mem)
        check_speedup_error(aev, aev_ref, delta, delta_ref)

        if parser.nnpops_cu_out and file.endswith(".pdb"):
            print('NNPops (C++ directly):')
            commands = f"./{parser.nnpops_cu_out} molecules/{file} {N}"
            subprocess.run(commands, shell=True, check=True, universal_newlines=True)
            print()

        print('CUaev (Kamesh):')
        nnp.aev_computer.use_cuda_extension = True
        cuaev_computer = nnp.aev_computer
        aev, delta = benchmark(speciesPositions, cuaev_computer, N, check_gpu_mem)
        check_speedup_error(aev, aev_ref, delta, delta_ref)

        print('-'*70 + '\n')
