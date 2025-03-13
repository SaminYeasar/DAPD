# Efficient Reinforcement Learning by Discovering Neural Pathways


This repository contains the official implementation of **Data Adaptive Pathway Discovery (DAPD)** presented in [Efficient Reinforcement Learning by Discovering Neural Pathways](https://openreview.net/pdf?id=WEoOreP0n5).

If you use this code for your research, please consider citing the paper:

```
@inproceedings{arnob2024DAPD,
 author = {Arnob, Samin Yeasar and Ohib, Riyasat and Plis, Sergey and Zhang, Amy and Sordoni, Alessandro and Precup, Doina},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {18660--18694},
 publisher = {Curran Associates, Inc.},
 title = {Efficient Reinforcement Learning by Discovering Neural Pathways},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/216f4cd12cfd69d46770bb2b491ae24b-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
---

## How to run the code:

### Install dependencies
* Create environment: `conda env create -f environment.yaml`
* Activate Environment: `conda activate DAPD`

---
### Train online single task
* We are using [Soft-Actor-Critic](https://arxiv.org/pdf/1801.01290) algorithm for our online RL single task experiments and PyTorch implementation from https://github.com/denisyarats/pytorch_sac

* `Directory: Online\`

* **DAPD**
  * `python main.py env="HalfCheetah-v2" algo="DAPD" keep_ratio=0.05 iterative_pruning=True continual_pruning=False ips_threshold=8000 mask_update_mavg=1`
  * **NOTE**: To reproduce results, Use the hyper-parameters mentioned in the paper
````
Important Hyper-parameters for DAPD:
keep_ratio=0.05           # ratio (out of 1) of parameters we want to train. ex: keep_ratio=0.05 means we are pruning 95% of the network and keeping 5% for training
mask_update_mavg=1        # Length og moving average, K
iterative_pruning=True    # Allows periodic mask update
continual_pruning=False   # if True: Keep pruning all the way, if False: stop pruning after reaching threshold episodic return
ips_threshold=8000        # iterative pruning stopping (ips) when reached this threshold (TH) reward
````

### Baselines:
* **Dense**
  * `python main.py env="HalfCheetah-v2" algo="dense"  keep_ratio=1.0`
  
For sparse performance comparison we are using the RiGL and Rlx2 implementation presented at [Rlx2](https://github.com/tyq1024/RLx2).

* **[RiGL](https://proceedings.mlr.press/v162/graesser22a/graesser22a.pdf)**
  * `python main.py env="HalfCheetah-v2" algo="rigl"  keep_ratio=0.05`
* **[Rlx2](https://arxiv.org/pdf/2205.15043)**
  * `python main.py env="HalfCheetah-v2" algo="rlx2" keep_ratio=0.05 use_dynamic_buffer=True`


