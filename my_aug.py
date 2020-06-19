import albumentations as A
from PIL import Image
import cv2
import os

def preprocess_input(im):

    aug = A.Compose([
        #A.Flip(p=1),
        #A.RandomGamma(gamma_limit=(100, 200), p=0.5),
        A.RandomBrightnessContrast(p=1),
        #A.ShiftScaleRotate(rotate_limit=180, scale_limit=(0, 1), p=1),

        A.CoarseDropout(max_holes=10, min_holes=5, max_width=4, max_height=200, fill_value=0, p=1),
        A.CoarseDropout(max_holes=10, min_holes=5, max_width=200, max_height=4, fill_value=0, p=1),
        #A.Rotate(limit=180, p=1),
        #A.RGBShift(p=0.5),
        A.GaussNoise(p=1),
        # A.CenterCrop(height=100, width=100, p=1),
        # A.Normalize(p=1.0),
        #A.HueSaturationValue(p=0.5),
        #A.ChannelShuffle(p=0.5),
        A.MedianBlur(p=1, blur_limit=5),
    ])
    augmented = aug(image=im)
    return augmented['image']

im = cv2.imread(os.path.join(os.getcwd(), '512x512-test', '512x512-test', 'ISIC_0052349.jpg'))
aug = preprocess_input(im)

img = Image.fromarray(im, 'RGB')
img.show()
augment = Image.fromarray(aug, 'RGB')
augment.show()