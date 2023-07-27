import albumentations as A
from torchvision import transforms as T

def get_train_transforms(mean, std, p):

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

#
    return [dict(type='t', value=t), dict(type='a', value=a)]


def get_test_transforms(mean, std):
    # Test data transformations
    a = A.Compose([
        A.Normalize(mean, std)
    ])

    return [dict(type='a', value=a)]