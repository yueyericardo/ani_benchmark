import time
import torch
import torchani
import os
from ase.io import read

path = os.path.dirname(os.path.realpath(__file__))
ani1x = torchani.models.ANI1x(periodic_table_index=False, model_index=None)


def benchmark_batch(device, run_force):
    # load model
    model = ani1x.to(device)
    # read file
    filepath = os.path.join(path, 'test.xyz')
    mol = read(filepath)
    species = torch.tensor([mol.get_atomic_numbers()], device=device)
    positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=device)
    # repeat 20k conformations
    species = torch.repeat_interleave(species, 20000, dim=0)
    positions = torch.repeat_interleave(positions, 20000, dim=0).requires_grad_(run_force)
    # split data into batches
    batch_size = 1000
    species_batches = species.split(batch_size)
    positions_batches = positions.split(batch_size)
    num_batch = len(species_batches)
    # benchmark
    torch.cuda.synchronize()
    start_time = time.time()

    for i in range(num_batch):
        spe = species_batches[i]
        pos = positions_batches[i]
        spe_pos = model.species_converter((spe, pos))
        _, energies = model(spe_pos)
        if run_force:
            forces = -torch.autograd.grad(energies.sum(), positions, create_graph=True, retain_graph=True)[0]

    torch.cuda.synchronize()
    time_sec = (time.time() - start_time)
    print(f'{time_sec:02.3f} seconds')


devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
for device in devices:
    print(f'device: {device:<4}, energy       : ', end='')
    benchmark_batch(device, run_force=False)
    print(f'device: {device:<4}, energy+force : ', end='')
    benchmark_batch(device, run_force=True)
