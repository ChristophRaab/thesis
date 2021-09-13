# Source Code for Chapter 6
## Reactive Robust Soft Learning Vector Quantization  
<br />

Content: 

    ├── rrslvq                  # Code for chapter 6: Non-Stationary Online Prototype Learning
    │   ├── demo.py             # Demo for Reactive Robust Soft Learning Vector Quantization (RRSLVQ)
    │   ├── plots               # Folder of plot scripts
    │   ├── study               # Folder of study scripts
    └── └── └── study_<type>.m  # Reproduces experiments of type <type>   

<br />

### Demo
For a demo on one dataset execute:

```bash
python demo.py # Demo file
```
<br />

### Study results on benchmarks
Reproducing thesis results:
```bash
# Most important for the thesis
python study/study_performance_detectors.py # Reproduce results for concept drift detectors.
python study/study_performance_det.py # Reproduce results for RRSLVQ and competitive algorithms.
python study/study_time|memory.py # Reproduce time and memory results.
```
All parameters are set. To change parameters and datasets edit the scripts.
<br />
<br />
Pre computed results are available. To reproduce plots execute:
```bash
python plots/*.py # Select wished plots
```