import torch
import torchani
import time
import argparse
import pkbar
from NNPOps.SymmetryFunctions import TorchANISymmetryFunctions


def benchmark_aev(parser, dataset, aev_comp):
    print('=> start running')
    torch.cuda.synchronize()
    start = time.time()
    aev_result = []

    for epoch in range(0, parser.num_epochs):

        print('Epoch: %d/%d' % (epoch + 1, parser.num_epochs))
        progbar = pkbar.Kbar(target=len(dataset)-1, width=8, always_stateful=True)

        for i, properties in enumerate(dataset):
            species = properties['species'].to(parser.device)
            coordinates = properties['coordinates'].to(parser.device).float()
            _, aev = aev_comp((species, coordinates))
            if epoch == 0:
                aev_result.append(aev)
            progbar.update(i, values=[("molecule_atoms", species.shape[-1])])

    torch.cuda.synchronize()
    stop = time.time()

    total_time = (stop - start) / parser.num_epochs
    print('Time per epoch - {:.1f}s\n'.format(total_time))
    return(aev_result)


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path',
                        help='Path of the dataset, can a hdf5 file \
                            or a directory containing hdf5 files')
    parser.add_argument('-d', '--device',
                        help='Device of modules and tensors',
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-b', '--batch_size',
                        help='Number of conformations of each batch',
                        default=1, type=int)
    parser.add_argument('-n', '--num_epochs',
                        help='epochs',
                        default=2, type=int)
    parser = parser.parse_args()

    print('=> loading dataset...')
    shifter = torchani.EnergyShifter(None)
    dataset = torchani.data.load(parser.dataset_path, additional_properties=('forces',)).subtract_self_energies(shifter).species_to_indices()
    print('=> Caching shuffled dataset...')
    dataset_shuffled = list(dataset.shuffle().collate(parser.batch_size))

    nnp1 = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(parser.device)
    nnp2 = torchani.models.ANI2x(periodic_table_index=True, model_index=None).to(parser.device)
    nnp2.aev_computer.use_cuda_extension = True
    aev_computer = nnp1.aev_computer
    cuaev_computer = nnp2.aev_computer
    nnpops_computer = TorchANISymmetryFunctions(nnp1.aev_computer).to(parser.device)

    print('Original TorchANI symmetry functions')
    aevs_ref = benchmark_aev(parser, dataset_shuffled, aev_computer)
    print('Optimized TorchANI symmetry functions')
    aevs_nnpops = benchmark_aev(parser, dataset_shuffled, nnpops_computer)
    print('CUaev (Kamesh)')
    aevs_cuaev = benchmark_aev(parser, dataset_shuffled, cuaev_computer)

    for i, aev_ref in enumerate(aevs_ref):
        # print(i)
        aev_cuaev = aevs_cuaev[i]
        aev_nnpops = aevs_nnpops[i]
        cuaev_error = torch.max(torch.abs(aev_cuaev - aev_ref))
        # print(f'  cuaev Error: {cuaev_error:.1e}\n')
        assert cuaev_error < 1e-4, f'  cuaev Error: {cuaev_error:.1e}\n'
        nnpopsaev_error = torch.max(torch.abs(aev_nnpops - aev_ref))
        # print(f'  nnpops Error: {nnpopsaev_error:.1e}\n')
        assert nnpopsaev_error < 1e-4, f'  cuaev Error: {nnpopsaev_error:.1e}\n'
