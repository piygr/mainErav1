# Session 10 Assignment

1. ResNet architecture for CIFAR10 that has the following architecture:
    1. PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    2. Layer1 -
       - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
       - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
       - Add(X, R1)
    3. Layer 2 -
       - Conv 3x3 [256k]
       - MaxPooling2D
       - BN
       - ReLU
    4. Layer 3 -
       - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
       - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
       - Add(X, R2)
    5. MaxPooling with Kernel Size 4
    6. FC Layer 
    7. SoftMax
2. Uses One Cycle Policy such that:
    1. Total Epochs = 24
    2. Max at Epoch = 5
    3. LRMIN = FIND
    4. LRMAX = FIND
    5. NO Annihilation
3. Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
4. Batch size = 512
5. Use ADAM, and CrossEntropyLoss
6. Target Accuracy: 90%

------
## custom_resnet.py
The file contains the custom resnet model as desired in the assignment. 
Total number of trainable parameters = ~6.5M.

Here is the summary of the network -

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,792
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
         Dropout2d-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,856
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
         Dropout2d-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,584
      BatchNorm2d-11          [-1, 128, 16, 16]             256
        Dropout2d-12          [-1, 128, 16, 16]               0
           Conv2d-13          [-1, 128, 16, 16]         147,584
      BatchNorm2d-14          [-1, 128, 16, 16]             256
        Dropout2d-15          [-1, 128, 16, 16]               0
           Conv2d-16          [-1, 256, 16, 16]         295,168
        MaxPool2d-17            [-1, 256, 8, 8]               0
      BatchNorm2d-18            [-1, 256, 8, 8]             512
             ReLU-19            [-1, 256, 8, 8]               0
        Dropout2d-20            [-1, 256, 8, 8]               0
           Conv2d-21            [-1, 512, 8, 8]       1,180,160
        MaxPool2d-22            [-1, 512, 4, 4]               0
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
        Dropout2d-25            [-1, 512, 4, 4]               0
           Conv2d-26            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
        Dropout2d-28            [-1, 512, 4, 4]               0
           Conv2d-29            [-1, 512, 4, 4]       2,359,808
      BatchNorm2d-30            [-1, 512, 4, 4]           1,024
        Dropout2d-31            [-1, 512, 4, 4]               0
        MaxPool2d-32            [-1, 512, 1, 1]               0
           Linear-33                   [-1, 10]           5,130
================================================================
Total params: 6,575,370
Trainable params: 6,575,370
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 7.38
Params size (MB): 25.08
Estimated Total Size (MB): 32.47
----------------------------------------------------------------
```

## transforms.py
The file contains trasforms which are applied to the input dataset as per the assignment requirement.
```
t = T.Compose(
        [
            T.RandomCrop( (32, 32), padding=4, fill=(mean[0]*255, mean[1]*255, mean[2]*255) )
        ]
    )

    a = A.Compose(
        [
            A.Normalize(mean, std),
            A.HorizontalFlip(p=p),
            A.CoarseDropout(max_holes = 1,
                            max_height=8,
                            max_width=8,
                            min_holes = 1,
                            min_height=8,
                            min_width=8,
                            fill_value=mean,
                            mask_fill_value = None,
                            p=p
            )
        ]
    )
```

## dataset.py
CustomCIFAR10Dataset is created on top of CIFAR10 to take care of albumentation + torchvision transforms

## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S10.ipynb
The file is an IPython notebook.
The LRfinder has been used to find the Max LR. Multiple LR values were tried starting from 0.1 to 1e-8 to find that where the loss is lowest. Based on that the Max LR value is set to 3.20E-04.

**LRFinder**

<img width="705" alt="Screenshot 2023-07-14 at 11 39 09 PM" src="https://github.com/piygr/s10erav1/assets/135162847/20d5404e-d580-4b50-b246-39b1292c40ab">


```
Epoch 1 LR [3.2e-06]
Test set: Average loss: 1.4072, Accuracy: 4903/10000 (49.03%)

Epoch 2 LR [6.656000000000011e-05]
Test set: Average loss: 1.1207, Accuracy: 5979/10000 (59.79%)

Epoch 3 LR [0.0001299199999999999]
Test set: Average loss: 0.9486, Accuracy: 6603/10000 (66.03%)

Epoch 4 LR [0.00019328]
Test set: Average loss: 0.7799, Accuracy: 7267/10000 (72.67%)

Epoch 5 LR [0.0002566400000000001]
Test set: Average loss: 0.7548, Accuracy: 7336/10000 (73.36%)

Epoch 6 LR [0.00031999999999999986]
Test set: Average loss: 0.6213, Accuracy: 7901/10000 (79.01%)

Epoch 7 LR [0.0003033263157894737]
Test set: Average loss: 0.6122, Accuracy: 7897/10000 (78.97%)

Epoch 8 LR [0.0002866526315789474]
Test set: Average loss: 0.5619, Accuracy: 8130/10000 (81.30%)

Epoch 9 LR [0.0002699789473684211]
Test set: Average loss: 0.4941, Accuracy: 8326/10000 (83.26%)

Epoch 10 LR [0.0002533052631578948]
Test set: Average loss: 0.4339, Accuracy: 8541/10000 (85.41%)

Epoch 11 LR [0.0002366315789473684]
Test set: Average loss: 0.4452, Accuracy: 8496/10000 (84.96%)

Epoch 12 LR [0.00021995789473684213]
Test set: Average loss: 0.4931, Accuracy: 8367/10000 (83.67%)

Epoch 13 LR [0.0002032842105263158]
Test set: Average loss: 0.4949, Accuracy: 8346/10000 (83.46%)

Epoch 14 LR [0.00018661052631578954]
Test set: Average loss: 0.3910, Accuracy: 8699/10000 (86.99%)

Epoch 15 LR [0.0001699368421052631]
Test set: Average loss: 0.4118, Accuracy: 8601/10000 (86.01%)

Epoch 16 LR [0.00015326315789473684]
Test set: Average loss: 0.4488, Accuracy: 8537/10000 (85.37%)

Epoch 17 LR [0.0001365894736842106]
Test set: Average loss: 0.3487, Accuracy: 8860/10000 (88.60%)

Epoch 18 LR [0.00011991578947368415]
Test set: Average loss: 0.3302, Accuracy: 8902/10000 (89.02%)

Epoch 19 LR [0.0001032421052631579]
Test set: Average loss: 0.3080, Accuracy: 8962/10000 (89.62%)

Epoch 20 LR [8.656842105263165e-05]
Test set: Average loss: 0.3482, Accuracy: 8859/10000 (88.59%)

Epoch 21 LR [6.98947368421052e-05]
Test set: Average loss: 0.2918, Accuracy: 9047/10000 (90.47%)

Epoch 22 LR [5.3221052631578956e-05]
Test set: Average loss: 0.2971, Accuracy: 9041/10000 (90.41%)

Epoch 23 LR [3.6547368421052686e-05]
Test set: Average loss: 0.2621, Accuracy: 9159/10000 (91.59%)

Epoch 24 LR [1.9873684210526257e-05]
Test set: Average loss: 0.2544, Accuracy: 9170/10000 (91.70%)
```

## How to setup
### Prerequisits
```
1. python 3.8 or higher
2. pip 22 or higher
```

It's recommended to use virtualenv so that there's no conflict of package versions if there are multiple projects configured on a single system. 
Read more about [virtualenv](https://virtualenv.pypa.io/en/latest/). 

Once virtualenv is activated (or otherwise not opted), install required packages using following command. 

```
pip install requirements.txt
```

## Running IPython Notebook using jupyter
To run the notebook locally -
```
$> cd <to the project folder>
$> jupyter notebook
```
The jupyter server starts with the following output -
```
To access the notebook, open this file in a browser:
        file:///<path to home folder>/Library/Jupyter/runtime/nbserver-71178-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
     or http://127.0.0.1:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
```

Open the above link in your favourite browser, a page similar to below shall be loaded.

![Jupyter server index page](https://github.com/piygr/s5erav1/assets/135162847/40087757-4c99-4b98-8abd-5c4ce95eda38)

- Click on the notebook (.ipynb) link.

A page similar to below shall be loaded. Make sure, it shows *trusted* in top bar. 
If it's not _trusted_, click on *Trust* button and add to the trusted files.

![Jupyter notebook page](https://github.com/piygr/s5erav1/assets/135162847/7858da8f-e07e-47cd-9aa9-19c8c569def1)
Now, the notebook can be operated from the action panel.

Happy Modeling :-) 
 
