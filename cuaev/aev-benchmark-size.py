import time
import torch
import torchani
import pynvml
import gc
import os
from ase.io import read
import argparse
from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions

summary = '\n'
runcounter = 0
N = 200
last_py_speed = None


def checkgpu(device=None):
    i = device if device else torch.cuda.current_device()
    t = torch.cuda.get_device_properties(i).total_memory
    c = torch.cuda.memory_reserved(i)
    name = torch.cuda.get_device_properties(i).name
    print('   GPU Memory Cached (pytorch) : {:7.1f}MB / {:.1f}MB ({})'.format(c / 1024 / 1024, t / 1024 / 1024, name))
    real_i = int(os.environ['CUDA_VISIBLE_DEVICES'][0]) if 'CUDA_VISIBLE_DEVICES' in os.environ else i
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(real_i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    name = pynvml.nvmlDeviceGetName(h)
    print('   GPU Memory Used (nvidia-smi): {:7.1f}MB / {:.1f}MB ({})'.format(info.used / 1024 / 1024, info.total / 1024 / 1024, name.decode()))
    return f'{(info.used / 1024 / 1024):.1f}MB'


def alert(text):
    print('\033[91m{}\33[0m'.format(text))  # red


def info(text):
    print('\033[32m{}\33[0m'.format(text))  # green


def format_time(t):
    if t < 1:
        t = f'{t * 1000:.1f} ms'
    else:
        t = f'{t:.3f} sec'
    return t


def addSummaryLine(items=None, init=False):
    if init:
        addSummaryEmptyLine()
        items = ['RUN', 'PDB', 'Size', 'forward', 'backward', 'Others', 'Total', f'Total({N})', 'Speedup', 'GPU']
    global summary
    summary += items[0].ljust(23) + items[1].ljust(13) + items[2].ljust(13) + items[3].ljust(13) + items[4].ljust(13) + items[5].ljust(13) + \
        items[6].ljust(13) + items[7].ljust(13) + items[8].ljust(13) + items[9].ljust(13) + '\n'


def addSummaryEmptyLine():
    global summary
    summary += f"{'-'*23}".ljust(23) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + \
        f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + f"{'-'*13}".ljust(13) + '\n'


def benchmark(speciesPositions, aev_comp, runbackward=False, mol_info=None, verbose=True):
    global runcounter
    global last_py_speed
    mode = 'py'
    if hasattr(aev_comp, 'use_cuda_extension'):
        if aev_comp.use_cuda_extension:
            mode = 'cu'
        else:
            mode = 'py'
    else:
        mode = 'nnpops'
    mode_align = mode.ljust(6)
    runname = f"{mode_align} aev fd{'+bd' if runbackward else''}"
    items = [f'{(runcounter+1):02} {runname}', f"{mol_info['name']}", f"{mol_info['atoms']}", '-', '-', '-', '-', '-', '-', '-']

    forward_time = 0
    force_time = 0
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()
    start = time.time()

    aev = None
    force = None
    gpumem = None
    for i in range(N):
        species, coordinates = speciesPositions
        coordinates = coordinates.requires_grad_(runbackward)

        torch.cuda.synchronize()
        forward_start = time.time()
        try:
            _, aev = aev_comp((species, coordinates))
        except Exception as e:
            alert(f"  AEV faild: {str(e)[:50]}...")
            addSummaryLine(items)
            runcounter += 1
            return None, None, None
        torch.cuda.synchronize()
        forward_time += time.time() - forward_start

        if runbackward:  # backward
            force_start = time.time()
            try:
                force = -torch.autograd.grad(aev.sum(), coordinates, create_graph=True, retain_graph=True)[0]
            except Exception as e:
                alert(f" Force faild: {str(e)[:50]}...")
                addSummaryLine(items)
                runcounter += 1
                return None, None, None
            torch.cuda.synchronize()
            force_time += time.time() - force_start

        if i == 2 and verbose:
            gpumem = checkgpu()

    torch.cuda.synchronize()
    total_time = (time.time() - start) / N
    force_time = force_time / N
    forward_time = forward_time / N
    others_time = total_time - force_time - forward_time

    if verbose:
        if mode != 'py':
            if last_py_speed is not None:
                speed_up = last_py_speed / total_time
                speed_up = f'{speed_up:.2f}'
            else:
                speed_up = '-'
        else:
            last_py_speed = total_time
            speed_up = '-'

    if verbose:
        print(f'  Duration: {total_time * N:.2f} s')
        print(f'  Speed: {total_time*1000:.2f} ms/it')
        if runcounter == 0:
            addSummaryLine(init=True)
            addSummaryEmptyLine()
        if runcounter >= 0:
            items = [f'{(runcounter+1):02} {runname}',
                     f"{mol_info['name']}",
                     f"{mol_info['atoms']}",
                     f'{format_time(forward_time)}',
                     f'{format_time(force_time)}',
                     f'{format_time(others_time)}',
                     f'{format_time(total_time)}',
                     f'{format_time(total_time * N)}',
                     f'{speed_up}',
                     f'{gpumem}']
            addSummaryLine(items)
        runcounter += 1

    return aev, total_time, force


def check_speedup_error(aev, aev_ref, speed, speed_ref, force, force_ref):
    if (speed_ref is not None) and (speed is not None) and (aev is not None) and (aev_ref is not None):
        speedUP = speed_ref / speed
        if speedUP > 1:
            info(f'  Speed up: {speedUP:.2f} X\n')
        else:
            alert(f'  Speed up (slower): {speedUP:.2f} X\n')

        aev_error = torch.max(torch.abs(aev - aev_ref))
        assert aev_error < 0.02, f'  Error: {aev_error:.1e}\n'

    if (force is not None) and (force_ref is not None):
        force_error = torch.max(torch.abs(force - force_ref))
        assert force_error < 0.02, f'  Error: {force_error:.1e}\n'


def run(file, nnp_ref, nnp_cuaev, nnpops_computer, runbackward, maxatoms=10000):
    nnpops_computer = TorchANISymmetryFunctions(nnp_ref.aev_computer).to(device)
    filepath = os.path.join(path, f'molecules/{file}')
    mol = read(filepath)
    species = torch.tensor([mol.get_atomic_numbers()], device=device)
    positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=device)
    spelist = list(torch.unique(species.flatten()).cpu().numpy())
    species = species[:, :maxatoms]
    positions = positions[:, :maxatoms, :]
    speciesPositions = nnp_ref.species_converter((species, positions))
    print(f'File: {file}, Molecule size: {species.shape[-1]}, Species: {spelist}\n')

    if args.nsight:
        torch.cuda.nvtx.range_push(file)

    print('Original TorchANI:')
    mol_info = {'name': file, 'atoms': species.shape[-1]}
    aev_ref, delta_ref, force_ref = benchmark(speciesPositions, nnp_ref.aev_computer, runbackward, mol_info)
    print()

    print('CUaev:')
    # warm up
    _, _, _ = benchmark(speciesPositions, nnp_cuaev.aev_computer, runbackward, mol_info, verbose=False)
    # run
    aev, delta, force = benchmark(speciesPositions, nnp_cuaev.aev_computer, runbackward, mol_info)
    check_speedup_error(aev, aev_ref, delta, delta_ref, force, force_ref)
    print()

    print('NNPops:')
    # warm up
    _, _, _ = benchmark(speciesPositions, nnpops_computer, runbackward, mol_info, verbose=False)
    # run
    aev, delta, force = benchmark(speciesPositions, nnpops_computer, runbackward, mol_info)
    check_speedup_error(aev, aev_ref, delta, delta_ref, force, force_ref)

    global last_py_speed
    last_py_speed = None
    if args.nsight:
        torch.cuda.nvtx.range_pop()

    print('-' * 70 + '\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--nsight',
                        action='store_true',
                        help='use nsight profile')
    parser.add_argument('-b', '--backward',
                        action='store_true',
                        help='benchmark double backward')
    parser.add_argument('-n', '--N',
                        help='Number of Repeat',
                        default=200, type=int)
    parser.set_defaults(backward=0)
    args = parser.parse_args()
    path = os.path.dirname(os.path.realpath(__file__))
    N = args.N

    device = torch.device('cuda')
    files = ['small.pdb', '6W8H.pdb']
    # files = ['small.pdb']
    nnp_ref = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
    nnp_cuaev = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
    nnp_cuaev.aev_computer.use_cuda_extension = True
    nnpops_computer = None  # needed be initilize for every pdb

    if args.nsight:
        N = 3
        torch.cuda.profiler.start()

    for file in files:
        run(file, nnp_ref, nnp_cuaev, nnpops_computer, runbackward=False)
    # for maxatom in [6000, 10000, 13000]:
    for maxatom in [6000, 10000]:
        file = '1C17.pdb'
        run(file, nnp_ref, nnp_cuaev, nnpops_computer, runbackward=False, maxatoms=maxatom)

    addSummaryEmptyLine()
    info('Add Backward\n')

    for file in files:
        run(file, nnp_ref, nnp_cuaev, nnpops_computer, runbackward=True)
    for maxatom in [6000, 10000]:
        file = '1C17.pdb'
        run(file, nnp_ref, nnp_cuaev, nnpops_computer, runbackward=True, maxatoms=maxatom)
    addSummaryEmptyLine()

    print(summary)
    if args.nsight:
        torch.cuda.profiler.stop()
