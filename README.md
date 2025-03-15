# Dissertation Learning in Non-Stationary Environments - Code and Thesis
<br />
> For data please contact me
This repository contains the source code and the final pdf for the dissertation 'Learning in Non-Stationary Environments'.
Standard language is python and mentioned otherwise. The files are usually self-explaining for example, demo or train algorithms, utils, plot methods etc.
The folder structure roughly represents the chapters of the thesis:

    ├── sda                     # Code for chapter 3 & 4: Geometric and Subspace Domain Adaptation    
    │   ├── gda.py              # Python Demo for Geometric Domain Adaptation
    │   ├── so.py               # Python Demo for Subspace Override Algorithm
    │   ├── nso.py              # Python Demo for Nyström SO
    │   ├── matlab              # Matlab Code 
    │   │   ├── study_all.m     # Reproduces all performance experiments (takes a while!)   
    │   │   ├── study_<name>.m  # Reproduces performance experiments on dataset <name>   
    ├── dsda                    # Code for chapter 5: Deep Spectral Domain Adaptation
    │   ├── train_asan.py       # Demo for Adversarial Spectral Adaptation Network
    │   ├── train_dsn.py        # Demo for Deep Spectral Network
    │   ├── study_asan.py       # Reproduces ASAN performance experiments on selected dataset (takes a while!)   
    │   ├── study_dsn.py        # Reproduces DSN performance experiments on selected dataset (takes a while!)   
    ├── rrslvq                  # Code for chapter 6: Non-Stationary Online Prototype Learning
    │   ├── demo.py             # Demo for Reactive Robust Soft Learning Vector Quantization (RRSLVQ)
    │   ├── study               # Folder of study scripts
    └── └── └── study_<type>.m  # Reproduces experiments of type <type>   
    └── thesis_learning_in_non_stationary_enviroments.pdf       # Dissertation as Pdf  

<br />
<br />

## Installation
Use the requirements.txt and pip_requirements.txt to install dependencies. 
The recommended enviroment is conda in combindation with pip. 

```bash
git clone https://github.com/ChristophRaab/thesis.git 
cd thesis
conda create -n thesis python=3.8
conda activate thesis
conda install --file requirements.txt
pip install -r pip_requirements.txt # If conda requirements fail
```
<br />

## Execution
All python scripts can be started from top-level directory or in the file directory. <br />
All matlab scripts must be started in the file directory.
<br />
<br />

## Datasets
All datasets are included except the Visda dataset required for reproducing the results in the appendix. 
The Visda dataset can be found [here](https://paperswithcode.com/dataset/visda-2017).<br />
To use the Visda datasets copy the extracted folders train, validation, test to:  
        
        ├── dsda                    # Code for chapter 4: Deep Spectral Domain Adaptation
        │   ├── dataset             # Domain Adaptation dataset


## Licenses
Code [MIT](https://choosealicense.com/licenses/mit/)<br />
Thesis [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)<br />
