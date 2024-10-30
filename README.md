![Static Badge](https://img.shields.io/badge/Journal-Nature%20Machine%20Intelligence-blue?link=https%3A%2F%2Fwww.nature.com%2Farticles%2Fs42256-023-00736-z) [![DOI](https://zenodo.org/badge/681432757.svg)](https://zenodo.org/badge/latestdoi/681432757) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white) ![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

# Code for "Machine learning uncovers degradation pathways of perovskite light-emitting diode with multispectral imaging"

### Paper link
https://www.nature.com/articles/s42256-023-00736-z

### Releases

- Version 1.0 (For Python 3.6, Cuda 10.0, and TensorFlow 1.14.0) https://doi.org/10.5281/zenodo.8281088
- Version 2.0 (For Python 3.6, Cuda 11.x, and TensorFlow 2.6.2) https://doi.org/10.5281/zenodo.8417653 

## System requirements

### Software
- python 3.6.8
- h5py 3.1.0
- scipy 1.5.4
- joblib 0.14.1
- matplotlib 3.1.2
- tensorflow_gpu 2.6.2 (support Cuda 11.2/11.4)

### Hardware
- 32GB RAM PC Memory
- (Optional) 16GB GPU Memory

### Tested enviornment
- Linux, CentOS 7
- Windows 11, 22H2

## Installation guide

### Install Anaconda (15-30 min)
https://docs.anaconda.com/anaconda/install/

### Create and activate new enviornment in anaconda prompt (5-10 min)
```bash
conda create -n (anyname) python=3.6.8
```
```bash
conda activate (anyname)
```

### Register new enviornment for Jupyter Notebook
```bash
conda install ipykernel
```

### Install dependencies through pip
```bash
pip install h5py==3.1.0
pip install matplotlib==3.1.2
pip install scipy==1.5.4
pip install joblib==0.14.1
```

To install the CPU version of tensorflow:
```bash
pip install tensorflow==2.6.2
```

OR To install the GPU version (which require a working GPU with CUDA 11.2 or 11.4 with a corresponding cuDNN driver):
```bash
pip install tensorflow_gpu==2.6.2
```


## Demo and instruction to use

Launch Jupyter Notebook and locate this repository.
https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html

Run PANet_blind.ipynb, follow the instruction to reproduce the reported result. 
**The attached Demo.pdf is a complete run of the demo with full logs.**

## License

[Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)]
https://creativecommons.org/licenses/by-nc/4.0/
