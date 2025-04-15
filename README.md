[![arXiv](https://img.shields.io/badge/arXiv-2411.06447-B31B1B.svg)](https://arxiv.org/abs/2411.06447)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15021550.svg)](https://doi.org/10.5281/zenodo.15021550)


</div>
<a href='https://github.com/falex-aimri' target='_blank'>Alex Finkelstein</a> |
<a href='https://github.com/vnikale' target='_blank'>Nikita Vladimirov</a> |
<a href='https://github.com/cest-sources' target='_blank'>Moritz Zaiss</a> |
<a href='https://github.com/operlman' target='_blank'>Or Perlman</a>】
<div>
<a href='https://mri-ai.github.io/' target='_blank'>Momentum Lab, Tel Aviv University</a>
</div>

# Neural Bloch-McConnell Fitting (NBMF)
Why & what: *Quantitative molecular imaging extracts biophysical parameter maps by solving physics-based inverse problems. We employ AI-driven methods to address the challenges of dynamic complexity & ill-posedness, enabling rapid and robust estimation scaled to millions of brain voxels.*
## 
This repository shares the implementation of the methods described in our work **"Multi-Parameter Molecular MRI Quantification using Physics-Informed Self-Supervised Learning"**, presented in ISMRM24' ([abstract](https://archive.ismrm.org/2024/4478.html)) and published in Communications Physics: https://www.nature.com/articles/s42005-025-02063-8. Also available as a preprint on Arxiv, see [abstract](https://arxiv.org/abs/2411.06447), [html full-text](https://arxiv.org/html/2411.06447), [pdf full-text](https://arxiv.org/pdf/2411.06447). 

Main targets, and ideas/methods developed and implemented to achieve them: 
* ✨Speed✨: **Jax-based auto-differentiated numerical solutions** of Ordinary Differential Equations (ODE); towards **computationally feasible gradients** of forward physics models relfecting complex dynamics simulation. This facilitates fitting and estimation of physical constants underlying the ODE coefficients from observed data.
  > In our study, we address the Bloch-McConnell (BM) ODEs governing the dynamics of macroscopic magnetization vectors of multiple exchanging proton pools (of water, macromolecules, amide-groups). These dynamics occur during the complex saturation- and spin-lock based 
  > magnetization-preparation phase of imaging protocols of the emerging MRI modality of Semisolid Magnetization Transfer (ssMT) and Chemical Exchange Saturation Transfer (CEST), and have to be simulated repeatedly in full in the case of clinical, quantitative, and non-steady-state Magnetic Resonance Fingerprinting (MRF) imaging protocols (ssMT/CEST-MRF). The voxel-wise **estimation** towards brain maps of the concentrations of proton pools and (pH-encoding) rates of their chemical exchange (hence _quantitative molecular imaging_), presents an ill-posed **inverse problem**. The direct solution of the above by nonlinear-least-squares (NLLSQ) fitting has been elusive because of the computational challenges; this work makes such **gradient-based**  NLLSQ optimization practical for the first time, unlocking above **x1000** acceleration w.r.t existing fitting implementations s.a [BM_sim_fit](https://github.com/cest-sources/BM_sim_fit).
* The forward model acceleration (sans gradients) is useful for **rapid creation of synthetic signal dictionaries**,
  > as used in the standard MRF-based approaches - matching and supervised learning. Please see _dictionary_methods_ folder for our Jax&GPU-accelerated reimplementations of these, with speed-ups summarized in paper (Table 1), reproducible via the notebooks.
* ✨Robustness✨: Augmenting fitting by a joint **self-supervised** learning of a **neural-network estimator cycle-consistent with the physical model** exhibits interesting robustness properties, at a very low computational or accuracy cost, and enables real-time quantification with accuracy and physics-consistency guarantees.
  >  The NBMF provides smoother maps than purely voxelwise methods such as simple fitting or "dot-product" - the standard MRF approach of seeking best match in a dictionary of signals (hence "fingerprint"), reflecting a neural _regularizing prior_. However, this entails no significant compromise on the goodness of fit, i.e. no increase of the _modeling error_ - the discrepancy between measured and modeled signal. Even more strikingly, the transfer of the NBMF-trained on a mere single subject network to real-time quantification of new data exhibits a much better _modeling error_ (in terms of estimates' consistency with signal under the assumed physics) compared to those observed for networks trained by _supervised learning_ on paired synthetic data. This calls into question the reliability of the rapid MRF decoding using supervised learning and motivates the exploration of self-supervised approaches in molecular MRF, anatomical MRF and inverse problems in general.


## ⚡ Getting Started

Clone the repo or download zip archive (in that case skip the next section, data files are included in the zip).

### Obtaining data files

Install git-lfs on your system by ```sudo apt-get install git-lfs``` or if you're not admin, by ```bash data/no_sudo_install_git_lfs.sh```

Then cd to the cloned repo folder, and run:
```bash
git lfs install
git lfs pull
```
The larger files in data will turn from pointers (~10K)  to actual files (>10M). 

### Demo of the methods, reproducing figures

The notebooks ```demo_in_vitro.ipynb``` and ```demo_in_vivo.ipynb``` can be viewed on web

In order to run them on the provided in-vitro and sample in-vivo data, please follow the instructions below.

### Setting up the environment

For the fast speeds reported, the workstation must have a GPU installed with appropriately updated drivers and CUDA middleware. Tested with NVIDIA GeForce RTX 3060, Driver Version: 550.120, CUDA Version: 12.4 .
The below assumes a Linux machine (we tested on Ubuntu22).

The environment can be created using conda with a single duplication command:
```bash
conda env create -f enviroment.yml
```

Alternatively, create and activate an environment using any suitable manager, e.g.:

(a) conda:
```
conda create -n nbmf --no-default-packages python=3.12
conda activate nbmf
```
(b) virtualenv:
```
virtualenv .venv
source .venv/bin/activate
```

and then in the target environment run:
```bash
pip install -r requirements.txt
```

## 🚀 Contributing
We believe in openly sharing information between research group and contribute data. 
Whether you have a question or a bug to fix, please let us know. See our group website at: https://mri-ai.github.io/

## 📑 References
If you use this code for research or software development please cite the following publication:
``` # TO CHANGE
Finkelstein, A., Vladimirov, N., Zaiss, M. et al. Multi-parameter molecular MRI quantification using physics-informed self-supervised learning. Commun Phys 8, 164 (2025). https://doi.org/10.1038/s42005-025-02063-8
```
