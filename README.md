[![DOI](https://zenodo.org/badge/681432757.svg)](https://zenodo.org/badge/latestdoi/681432757)

# Code demo for "Machine learning uncovers degradation pathways of perovskite LEDs with multispectral imaging"

## System requirements

### Software dependencies
- python 3.6.8
- h5py 3.1.0
- scipy 1.5.4
- joblib 0.14.1
- tensorflow_gpu 2.6.2 (support Cuda 11.2/11.4)
- matplotlib (any version)

### Tested enviornment
- Linux, CentOS 7
- Windows 11, 22H2

### Required hardware
- 32GB RAM PC Memory
- (Optional) 16GB GPU Memory

## Installation guide (Windows 10)

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

Launch Jupyter Notebook App and find the demo folder.
https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html

Run PANet_blind.ipynb, follow the instruction to reproduce the reported result. 
**The attached Demo.pdf is a complete run of the demo with full logs.**

## License

[Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)]
https://creativecommons.org/licenses/by-nc/4.0/