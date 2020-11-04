import mdtraj
import time
import torch
import torchani

from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions

device = torch.device('cuda')

mol = mdtraj.load('molecules/2iuz_ligand.mol2')
species = torch.tensor([[atom.element.atomic_number for atom in mol.top.atoms]], device=device)
positions = torch.tensor(mol.xyz, dtype=torch.float32, requires_grad=False, device=device)

nnp = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(device)
speciesPositions = nnp.species_converter((species, positions))
symmFuncRef = nnp.aev_computer
symmFunc = TorchANISymmetryFunctions(nnp.aev_computer).to(device)

N = 4000

aev_ref = symmFuncRef(speciesPositions).aevs
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    aev_ref = symmFuncRef(speciesPositions).aevs
torch.cuda.synchronize()
delta = time.time() - start
print('Original TorchANI symmetry functions')
print(f'  Duration: {delta:.2f} s')
print(f'  Speed: {delta/N*1000:.3f} ms/it')
print()


aev = symmFunc(speciesPositions).aevs
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    aev = symmFunc(speciesPositions).aevs
torch.cuda.synchronize()
delta = time.time() - start
print('Optimized TorchANI symmetry functions')
print(f'  Duration: {delta:.2f} s')
print(f'  Speed: {delta/N*1000:.3f} ms/it')
aev_error = torch.max(torch.abs(aev - aev_ref))
print(f'  Error: {aev_error:.1e}\n')
assert aev_error < 0.02


nnp.aev_computer.use_cuda_extension = True
cuaev_computer = nnp.aev_computer
torch.cuda.synchronize()
start = time.time()
for _ in range(N):
    aev = cuaev_computer(speciesPositions).aevs
torch.cuda.synchronize()
delta = time.time() - start
print('CUaev (Kamesh)')
print(f'  Duration: {delta:.2f} s')
print(f'  Speed: {delta/N*1000:.3f} ms/it')
aev_error = torch.max(torch.abs(aev - aev_ref))
print(f'  Error: {aev_error:.1e}   # note that this error is big, should be cuaevs bug to be fix\n')
assert aev_error < 0.02
