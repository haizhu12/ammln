# AMFF
AMFF
# Feature Fusion and Metric Learning Network for Zero-Shot Sketch-Based Image Retrieval
![entropy-25-00502-g001](https://user-images.githubusercontent.com/93024130/232699853-715b279d-51d2-47f7-a576-fa7a51914085.png)


#`setup`

1.ubuntu >= 20.04 LTS

2.conda create -n AMFF python=3.8

3.conda activate AMFF

4.pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

5.python setup.py install

6.pip install pandas pytorch_metric_learning timn scikiy-image

7.pip install pretrainedmodels nltk

![开场图](https://user-images.githubusercontent.com/93024130/232704514-9df9aa0d-b9f4-4401-972d-dab5a0deed06.png)

# Data set directory structure
-data/  
>-sketchy/   

>>-train/

>>>-photo/

>>>>-image1

>>>-sketch/

>>>>-image1   

>>-val/

>>>-photo/

>>>>-image1

>>>-sketch/

>>>>-image1
# Train Model 

## para 

--proj_dim 512 

--backbone_type resnet50 --data_name sketchy 

--tri_lambda 1 --batch_size 64 

--num-workers 8 --loss mathm --epochs 10

# dataset
[sketchy and tu-berlin](https://drive.google.com/drive/folders/1lce41k7cGNUOwzt-eswCeahDLWG6Cdk0?usp=sharing)
