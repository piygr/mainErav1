# Main ERAV1
Main Repo for all future models

## [models]
The folder contains all the models, currently resnet.py  
ResNet18 Trainable Parameters count = 11,173,962

## transforms.py
The file contains trasforms which are applied to the input dataset as per the assignment requirement. eg.
```
t = T.Compose(
        [
            T.RandomCrop( (32, 32), padding=4, fill=(mean[0]*255, mean[1]*255, mean[2]*255) )
        ]
    )

    a = A.Compose(
        [
            A.Normalize(mean, std),
            #A.HorizontalFlip(p=p),
            A.CoarseDropout(max_holes = 1,
                            max_height=16,
                            max_width=16,
                            min_holes = 1,
                            min_height=16,
                            min_width=16,
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
Few of the utility functions are - 
- plot_dataset_sample (to get the sense of the dataset)
- plot_incorrect_preds (to plot the misclassified images)
- plot_grad_cam ( to plot GradCam on the predicted images)
- train
- test
- plot_model_performance (to plot the model performance training as well as validation)
- create_model_checkpoint (to create a checkpoint to resume it later)
- load_model_from_checkpoint (to load an existing checkpoint)


## How to setup on Google Colab
In google colab notebook, do git clone of the repo

```
!git clone https://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@github.com/piygr/s11erav1.git
```

Happy Modeling :-) 
 
