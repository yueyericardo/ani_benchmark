# aev_benchmark 

benchmark of [NNPOps](https://github.com/peastman/NNPOps) and [CUaev](https://github.com/akkamesh/torchani/tree/enh-ext-aev/torchani/extension)

### cuaev install guide:
```bash
git clone git@github.com:akkamesh/torchani.git
cd torchani
pip install -e .
cd torchani/extension/
# if encounter error, you may need to install pytorch by 
# conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit={CUDA_VERSION} -c pytorch
pip install -e .
```

### Benchmark
```
pip install pkbar h5py
```
```
python benchmark1.py 
python benchmark2.py datas/
```
