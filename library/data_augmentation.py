# -*- coding: utf-8 -*-

import os
import glob
import cv2
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, Crop, Rotate


#define paths
dir = os.getcwd()
parent_dir = os.path.dirname(dir)

PATH = parent_dir + '/data/fat_muscle'                             
PATH_IMAGES = PATH + '/images/'
PATH_MASKS = PATH + '/masks/'
PATH_PREDICTIONS = PATH + '/predictions' 

import sys
sys.path.append(parent_dir)


def load_data(path_images, path_masks, split=0.1):

    images = sorted(glob(os.path.join(path_images, "*"))) #path to ultrasound images
    masks = sorted(glob(os.path.join(path_masks, "*"))) #path to masks

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def load_augmented_x_train_y_train(path):
    x_train = sorted(glob(os.path.join(path, 'augmented_data/images/*')))     
    y_train = sorted(glob(os.path.join(path, 'augmented_data/masks/*')))
    return x_train, y_train

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_data(images, masks, save_path, augment=True):
    H = 256
    W = 256

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.replace('\\', '/').split("/")[-1].split(".")

        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.replace('\\', '/').split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:     

            aug = RandomRotate90(p=1.0)  # p= probability, that method is applied to image 1 = 100%
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(num_steps=7, distort_limit=0.5, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,rotate_limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = RandomBrightnessContrast(brightness_limit=(-0.5, 1.5), contrast_limit=0.2, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']
            
            aug = Crop(p=1.0, x_min=0, y_min=0, x_max=529, y_max=259)
            augmented = aug(image=x, mask=y)
            x6 = augmented['image']
            y6 = augmented['mask']

            save_images = [x, x1, x2, x3, x4, x5, x6]
            save_masks =  [y, y1, y2, y3, y4, y5, y6]

        else:
            save_images = [x]
            save_masks = [y]

        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(images) == 1:
                tmp_img_name = f"{image_name}.{image_extn}"
                tmp_mask_name = f"{mask_name}.{mask_extn}"

            else:
                tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"
            image_path = os.path.join(save_path, "images", tmp_img_name)
            mask_path = os.path.join(save_path, "masks", tmp_mask_name)
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1

if __name__ == "__main__":
    """ Loading original images and masks. """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path_images = PATH_IMAGES, path_masks = PATH_MASKS,split = 0.1)
    print(f"Train Images: {len(train_x)} - Train Masks: {len(train_y)}")
    print(f"Valid Images: {len(valid_x)} - Valid Masks: {len(valid_y)}")
    print(f"Test Images: {len(test_x)} - Test Masks: {len(test_y)}")

    """ Creating folders for augmented images & masks. """
    create_dir(PATH + '/augmented_data/images')
    create_dir(PATH + '/augmented_data/masks')

    
    """ Applying data augmentation. """
    augment_data(train_x, train_y, PATH + '/augmented_data/', augment=True)

    
    """ Loading augmented images and masks. """
    train_x, train_y = load_augmented_x_train_y_train(PATH)
    print(f"Augmented Images: {len(train_x)} - Augmented Masks: {len(train_y)}")