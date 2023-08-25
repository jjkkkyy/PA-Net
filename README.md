[![DOI](https://zenodo.org/badge/681432757.svg)](https://zenodo.org/badge/latestdoi/681432757)
# Physics-aware noise-level-aware image restoration network (PANet)
Code demo for the paper "Machine learning uncovers degradation pathways of perovskite LEDs with multispectral imaging"

## System requirements

### Software dependencies
- python 3.6.8
- h5py 2.10.0
- tensorflow 1.14.0
- keras 2.3.1
- scipy 1.4.1
- matplotlib 3.1.2
- joblib 0.14.1

### Tested enviornment
Linux, CentOS 7
Windows 10, version 21H1

### Required hardware
32GB RAM PC Memory

## Installation guide (Windows 10)

### Install Anaconda (15-30 min)
https://docs.anaconda.com/anaconda/install/

### Create and activate new enviornment in anaconda prompt (5-10 min)
```bash
conda create -n demo python=3.6.8
```
```bash
conda activate demo
```

### Register new enviornment for Jupyter Notebook
```bash
conda install ipykernel
```

### Install dependencies through pip
```bash
pip install h5py==2.10.0
pip install keras==2.3.1
pip install scipy==1.4.1
pip install matplotlib==3.1.2
pip install joblib==0.14.1
```

To install the CPU version of tensorflow:
```bash
pip install tensorflow==1.14.0
```

OR To install the GPU version (which require a working GPU with CUDA 10.0 and cuDNN 7.4 driver):
```bash
pip install tensorflow-gpu==1.14.0
```

Install Tensorflow GPU with CUDA 10.0 and cudNN 7.4 for Python on Windows 10: https://medium.com/@hippiedev/install-tensorflow-gpu-with-cuda-10-0-and-cudnn-7-4-for-python-on-windows-10-be95629e4f54

## Demo and instruction to use

Launch Jupyter Notebook App and find the demo folder.
https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html

Run PANet_blind.ipynb, follow the instruction to reproduce the reported result. 
**The attached Demo.pdf is a complete run of the demo with full logs.**

## License

[Creative Commons Attribution-NonCommercial (CC BY-NC 4.0)]
https://creativecommons.org/licenses/by-nc/4.0/