# Source Code for Chapter 3 & 4
## Shallow Domain Adaptation  
<br />

Content: 

    ├── sda                         # Code for chapter 3 & 4: Shallow Domain Adaptation.
    │   ├── matlab                  # Matlab code for chapter 3 & 4: Preferable for reproducing results.
    │   │   ├── study_all.m         # Reproduces all results in these chapters.
    │   │   ├── study_<dataset>.m   # Reproduces results for specific <dataset>.
    │   └── └── plots               # Folder containing all plot scripts.
    │   ├── gda.py                  # Demo for Geometric Domain Adaptation (GDA).
    │   ├── so.py                   # Demo for Subspace Override (SO).
    └── └──nso.py                   # Demo for Nyström Subspace Override (NSO & cNSO).

<br />

### Demo
For a demo on one preselected dataset execute:

```bash
python gda.py                  # Demo for Geometric Domain Adaptation (GDA).
python so.py                   # Demo for Subspace Override (SO).
pythoon nso.py                  # Demo for Nyström Subspace Override (NSO & cNSO).
```
<br />

### Study results on benchmarks
For reproducing thesis results matlab is required:
```bash
# Most important for the thesis
matlab study_all.m         # Reproduces all results in these chapters.
matlab study_<dataset>.m   # Reproduces results for specific <dataset>.
```
All parameters are set. To change parameters and datasets edit the scripts.
<br />
<br />
To reproduce plots execute:
```bash
matlab plots/*.py # Select wished plots
```