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

First, get the Anaconda installer and install Anaconda (in `/tmp/anaconda3`)
using

```bash
cd /tmp/
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
# specify /tmp/anconda3 as your installation path
source /tmp/anaconda3/bin/activate
```

Second, we install PyTorch (v1.4) using

```bash
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Third, we install ``torchph``. 

```bash
pip install git+https://github.com/c-hofer/torchph.git@0.0.0
```

Fourth, we clone this GitHub repository and start a notebook server, using

```bash
cd /tmp/
git clone https://github.com/c-hofer/jmlr_2019.git --recurse-submodules
jupyter notebook
```