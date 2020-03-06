
This repository contains the code to reproduce the experiments of 

```
@article{Hofer19b,
    author    = {C.~Hofer, R.~Kwitt, and M.~Niethammer},
    title     = {Learning Representations of Persistence Barcodes Homology},
    booktitle = {JMLR},
    year      = {2019}}
```

The `core` folder contains some utility code while the actual training/testing code is in the top-level jupyter notebooks, which are named after the corresponding datasets. 

Installation
============

The setup was tested with the following system configuration:

- Ubuntu 18.04.2 LTS
- CUDA 10.1 (driver version 418.87.00)
- Anaconda (Python 3.7)
- PyTorch 1.4

In the following, we assume that we work in `/tmp` (obviously, you have to
change this to reflect your choice and using `/tmp` is, of course, not
the best choice :).

1. Get the Anaconda installer and install Anaconda (in `/tmp/anaconda3`)
using

```bash
cd /tmp/
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
# specify /tmp/anconda3 as your installation path
source /tmp/anaconda3/bin/activate
```

2. Install PyTorch (v1.4)

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

3. Install ``torchph``. 

```bash
pip install git+https://github.com/c-hofer/torchph.git@0.0.0
```

4. Clone this GitHub repository.

```bash
cd /tmp/
git clone https://github.com/c-hofer/jmlr_2019.git --recurse-submodules
```

5. Download data

All data can be downloaded [here](https://drive.google.com/open?id=148hoKBu1bbnWcAf4pErGWaOwnXzr7jxy). Unzip the ZIP file using `unzip`

```
cd /tmp/jmlr_2019/core
unzip jmlr2019_datasets.zip
```

This should create a folder `datasets` in `/tmp/jmlr_2019/core/`.

6. Start jupyter notebook server in repository folder.

```bash
cd /tmp/jmlr_2019
jupyter notebook
``` 
