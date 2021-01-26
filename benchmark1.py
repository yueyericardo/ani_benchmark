import time
import torch
import torchani
import pynvml
import gc
import os
import argparse
from ase.io import read
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


# @snoop()
def benchmark(species, positions, cell, pbc, aev_comp, N, check_gpu_mem, check_grad, check_energy):
    speciesPositions = nnp.species_converter((species, positions))
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    if check_energy:
        nnp1 = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
        nnp1.aev_computer = aev_comp
    start = time.time()

    for i in range(N):
        if not check_energy:
            aev = aev_comp(speciesPositions, cell, pbc).aevs
            if i == 2 and check_gpu_mem:
                checkgpu()
            if check_grad:
                sum_aev = torch.sum(aev)
                if positions.grad is not None:
                    positions.grad.zero_()
                sum_aev.backward()
                grad = positions.grad.clone()
            else:
                grad = None
        else:
            # TODO
            # speciesPositions = nnp1.species_converter((species, positions))
            species_aevs = aev_comp(speciesPositions, cell, pbc)
            # aevs = []
            # for nn in nnp1.neural_networks[5:8]:
            #     aevs.append(nn(species_aevs)[1])
            # aev = torch.cat(aevs, dim=0).sum(dim=0, keepdim=True)
            aev = nnp1.neural_networks(species_aevs)[1]
            if i == 2 and check_gpu_mem:
                checkgpu()
            if check_grad:
                sum_aev = torch.sum(aev)
                if positions.grad is not None:
                    positions.grad.zero_()
                sum_aev.backward()
                grad = positions.grad.clone()
            else:
                grad = None

    torch.cuda.synchronize()
    delta = time.time() - start
    # print(f'  Duration: {delta:.2f} s')
    print(f'  Speed: {delta/N*1000:.2f} ms/it')
    return aev, delta, grad


def check_speedup_error(aev, aev_ref, speed, speed_ref, grad=None, grad_ref=None):
    speedUP = speed_ref / speed
    if speedUP > 1:
        info(f'  Speed up: {speedUP:.2f} X\n')
    else:
        alert(f'  Speed up (slower): {speedUP:.2f} X\n')

    aev_error = torch.max(torch.abs(aev - aev_ref))
    assert aev_error < 0.02, f'  AEV Error: {aev_error:.1e}\n'
    if grad is not None and grad_ref is not None:
        grad_error = torch.max(torch.abs(grad - grad_ref))
        assert grad_error < 0.02, f'  Grad Error: {grad_error:.1e}\n'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--nnpops_cu_out',
                        default=None)
    parser.add_argument('-m', '--check_gpu_mem',
                        dest='check_gpu_mem',
                        action='store_const',
                        const=1)
    parser.add_argument('-g', '--check_grad',
                        dest='check_grad',
                        action='store_const',
                        const=1)
    parser.add_argument('-e', '--check_energy',
                        dest='check_energy',
                        action='store_const',
                        const=1)
    parser.add_argument('-p', '--pbc',
                        dest='use_pbc',
                        action='store_const',
                        const=1)
    parser.set_defaults(check_gpu_mem=0)
    parser.set_defaults(check_grad=0)
    parser.set_defaults(use_pbc=0)
    parser.set_defaults(check_energy=0)
    parser = parser.parse_args()

    check_gpu_mem = parser.check_gpu_mem
    check_grad = True if parser.check_grad else False
    check_energy = True if parser.check_energy else False
    print(f'Check Gradient: {check_grad}')

    device = torch.device('cuda')
    # files = ['2iuz_ligand.mol2', '1hvk_ligand.mol2', '2iuz_ligand.mol2', '3hkw_ligand.mol2', '3hky_ligand.mol2', '3lka_ligand.mol2', '3o99_ligand.mol2', 'small.pdb', '1hz5.pdb', '6W8H.pdb']
    files = ['small.pdb', '3NIR.pdb', '6W8H.pdb']
    if (parser.use_pbc):
        files = ['3NIR.pdb']

    for file in files:
        mol = read(f'molecules/{file}')
        species = torch.tensor([mol.get_atomic_numbers()], device=device)
        positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=check_grad, device=device)
        cell = torch.tensor(mol.get_cell(complete=True), dtype=torch.float32, device=device)
        pbc = torch.tensor(mol.get_pbc(), dtype=torch.bool, device=device)
        if pbc[0] is False or not parser.use_pbc:
            pbc = None
            cell = None
        print(f'File: {file}, Molecule size: {species.shape[-1]}\n')
        print(f'cell: {cell}')
        print(f'pbc: {pbc}')

        nnp = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
        symmFuncRef = nnp.aev_computer
        symmFuncRef.use_cuda_extension = False
        symmFunc = TorchANISymmetryFunctions(nnp.aev_computer).to(device)

        N = 200

        print('Original TorchANI:')
        aev_ref, delta_ref, grad_ref = benchmark(species, positions, cell, pbc, symmFuncRef, N, check_gpu_mem, check_grad, check_energy)
        print()

        print('NNPops:')
        aev, delta, grad = benchmark(species, positions, cell, pbc, symmFunc, N, check_gpu_mem, check_grad, check_energy)
        check_speedup_error(aev, aev_ref, delta, delta_ref, grad, grad_ref)

        if (not pbc):
            print('CUaev (Kamesh):')
            nnp.aev_computer.use_cuda_extension = True
            cuaev_computer = nnp.aev_computer
            aev, delta, grad = benchmark(species, positions, cell, pbc, cuaev_computer, N, check_gpu_mem, check_grad, check_energy)
            check_speedup_error(aev, aev_ref, delta, delta_ref)

        print('-' * 70 + '\n')
