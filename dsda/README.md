# Source Code for Chapter 5
 
## Prerequisites
Use conda and pip to install requirements

## Deep Spectral Network (DSN)  

## Adversarial Spectral Adaptation Networks (ASAN)
All the parameters are set to optimal in our experiments. The following are the command for each task. The test_interval can be changed, which is the number of iterations between near test.

```
Office-31

python train_image.py --gpu_id id --net ResNet50 --dset office --test_interval 500 --s_dset_path ../data/office/amazon_list.txt --t_dset_path ../data/office/webcam_list.txt
python train_image.py --gpu_id 0 --net ResNet50 --dset office --test_interval 500 --s_dset_path data/amazon.txt --t_dset_path data/webcam.txt --tl SLSSO --sn True
```
```
Office-Home

pythonn train_image.py --gpu_id id --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ../data/office-home/Art.txt --t_dset_path ../data/office-home/Clipart.txt
```
```
VisDA 2017

pythonn train_image.py --gpu_id id --net ResNet50 --dset visda --test_interval 5000 --s_dset_path ../data/visda-2017/train_list.txt --t_dset_path ../data/visda-2017/validation_list.txt
```
```
Image-clef

pythonn train_image.py --gpu_id id --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ../data/image-clef/b_list.txt --t_dset_path ../data/image-clef/i_list.txt
```

