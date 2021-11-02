## Tutorial: Predicting Phonon DoS with Euclidean Neural Networks
### 2021 MRS Fall Meeting

### Installation
1. Clone the repository:
`git clone https://github.com/ninarina12/phononDoS_tutorial.git`

`cd phononDoS_tutorial`

2. Create a virtual environment for the project:
`conda create -n pdos python=3.9`

`conda activate pdos`

3. Install necessary packages:
`pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html`

where `${TORCH}` and `${CUDA}` should be replaced by the specific CUDA version (e.g. `cu102`) and PyTorch version (e.g. `1.10.0`), respectively. For example:

`pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html`

### References
Mario Geiger, Tess Smidt, Alby M., Benjamin Kurt Miller, et al. Euclidean neural networks: e3nn (2020) v0.3.3. https://doi.org/10.5281/zenodo.5292912.