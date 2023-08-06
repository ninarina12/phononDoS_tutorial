[![DOI](https://zenodo.org/badge/423683286.svg)](https://zenodo.org/badge/latestdoi/423683286)
## Tutorial: Predicting phonon DoS with Euclidean neural networks
### 2021 MRS Fall Meeting | Symmetry-aware neural networks for the material sciences

This tutorial is presented through an interactive Jupyter notebook. We invite you to follow along with the code examples through either of the two options below:

#### 1. Run in Google Colaboratory
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ninarina12/phononDoS_tutorial/blob/main/phononDoS_colab.ipynb)

If you don't have access to a GPU or simply want to try out the code before installing anything locally, click the Colab badge above to run the notebook in Google Colaboratory. Package imports are handled within the notebook.

#### 2. Work from a local installation
To work from a local copy of the code:

1. Clone the repository:
	> `git clone https://github.com/ninarina12/phononDoS_tutorial.git`

	> `cd phononDoS_tutorial`

2. Create a virtual environment for the project:
	> `conda create -n pdos python=3.9`

	> `conda activate pdos`

3. Install all necessary packages:
	> `pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html`

	where `${TORCH}` and `${CUDA}` should be replaced by the specific CUDA version (e.g. `cpu`, `cu102`) and PyTorch version (e.g. `1.10.0`), respectively. For example:

	> `pip install -r requirements.txt -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html`

4. Run `jupyter notebook` and open `phononDoS.ipynb`.

### References
**Publication:** Zhantao Chen, Nina Andrejevic, Tess Smidt, *et al.* "Direct Prediction of Phonon Density of States With Euclidean Neural Networks." Advanced Science (2021): 2004214. https://onlinelibrary.wiley.com/doi/10.1002/advs.202004214

**E(3)NN:** Mario Geiger, Tess Smidt, Alby M., Benjamin Kurt Miller, *et al.* Euclidean neural networks: e3nn (2020) v0.4.2. https://doi.org/10.5281/zenodo.5292912.

**Dataset:** Guido Petretto, Shyam Dwaraknath, Henrique P. C. Miranda, Donald Winston, *et al.* "High-throughput Density-Functional Perturbation Theory phonons for inorganic materials." (2018) figshare. Collection. https://doi.org/10.6084/m9.figshare.c.3938023.v1
