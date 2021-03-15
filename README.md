# dissertation
This repository contains the source code and the final pdf for the dissertation 'Learning in Non-Stationary Environments'.
Standard language is python and mentioned otherwise.
The folder structure roughly represents the chapters of the thesis:

    ├── sda                     # Code for chapter 2 & 3: Geometric and Subspace Domain Adaptation    
    │   ├── gda.py              # Python Demo for Geometric Domain Adaptation
    │   ├── so.py               # Python Demo for Subspace Override Algorithm
    │   ├── nso.py              # Python Demo for Nyström SO
    │   ├── matlab              # Matlab Code 
    │   │   ├── study_all.m     # Reproduces all performance experiments (takes a while!)   
    │   │   ├── study_<name>.m  # Reproduces performance experiments on dataset <name>   
    ├── dsda                    # Code for chapter 4: Deep Spectral Domain Adaptation
    │   ├── train_image.py      # Demo for Adversarial Spectral Adaptation Network
    │   ├── study.py            # Reproduces performance experiments on selected dataset (takes a while!)   
    ├── rrslvq                  # Code for chapter 5: Non-Stationary Online Prototype Learning
    │   ├── demo.py             # Demo for Reactive Robust Soft Learning RSLVQ
    │   ├── study               # Folder of study scripts
    └── └── └── study_<type>.m  # Reproduces experiments of type <type>   

Remaining files are self-explaining algorithms, utils, plot methods etc.

## Installation
Use the requirements.txt to install dependencies. 
The recommended enviroment is conda in combindation with pip. 

```bash
conda create -n thesis python=3.8
conda activate thesis
conda install --file requirements.txt
```

## Datasets
All datasets are included except following: 
### Office-31
Office-31 dataset can be found [here](https://drive.google.com/file/d/11nywfWdfdBi92Lr3y4ga2Cu4_-FpWKUC/view?usp=sharing).

### Office-Home
Office-Home dataset can be found [here](https://drive.google.com/file/d/1W_U8GsILKdMSxqhnmTbYaaWhvQ-P4RJ1/view?usp=sharing).

### Image-clef
Image-Clef dataset can be found [here](https://drive.google.com/file/d/1lu1ouDoeucW8MgmaKVATwNt5JT7Uvk62/view?usp=sharing).

## License
[MIT](https://choosealicense.com/licenses/mit/)
