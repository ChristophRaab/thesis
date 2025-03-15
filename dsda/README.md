# Source Code for Chapter 5
<br />

## Deep Spectral Network (DSN)  
```bash
python train_dsn.py # Demo 
python study_dsn.py # Reproduce thesis results 
```
To change parameters and datasets edit the scripts. 
<br />
<br />
<br />


## Adversarial Spectral Adaptation Networks (ASAN)
All the parameters are set to optimal in our experiments. The following shows the command to run the demo on each task. The test_interval can be changed, which is the number of iterations between each test.

```bash
#Office-31
python train_asan.py --gpu_id id --net ResNet50 --dset office --test_interval 500 --s_dset_path data/office/amazon_list.txt --t_dset_path data/office/webcam_list.txt
```
```bash
#Office-Home
python train_asan.py --gpu_id id --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path data/office-home/Art.txt --t_dset_path data/office-home/Clipart.txt
```
```bash
#Image-clef
python train_asan.py --gpu_id id --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path data/image-clef/b_list.txt --t_dset_path data/image-clef/i_list.txt
```
```bash
#VisDA 2017
python train_asan.py --gpu_id id --net ResNet50 --dset visda --test_interval 5000 --s_dset_path data/visda-2017/train_list.txt --t_dset_path data/visda-2017/validation_list.txt
```

To reproduce the results in the thesis change file from ```train_asan.py``` to ```study_asan.py```. Parameters are the same.