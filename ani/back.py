def benchmark_single(device, run_force):
    # model
    model = ani1x.to(device)
    # read file
    filepath = os.path.join(path, 'test.xyz')
    mol = read(filepath)
    species = torch.tensor([mol.get_atomic_numbers()], device=device)
    positions = torch.tensor([mol.get_positions()], dtype=torch.float32, requires_grad=False, device=device)
    spe_pos = model.species_converter((species, positions))
    # benchmark 20k times
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(20000):
        _, energies = model(spe_pos)
        if run_force:
            forces = -torch.autograd.grad(energies.sum(), positions, create_graph=True, retain_graph=True)[0]
    torch.cuda.synchronize()
    time_sec = (time.time() - start_time)
    print(time_sec)