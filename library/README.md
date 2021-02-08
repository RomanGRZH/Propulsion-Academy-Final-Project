## Library
Welcome to the library of this project. This folder includes all the needed scripts to execute the models:

## Models
[models.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/models.py): This script includes all the models. Binary and multiclasss

## Metrics
[metrics.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/metrics.py): This script includes all the metrics used for this project.

## Data Augmentation
[data_augmentation.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/data_augmentation.py): Run this script to create new augmented data to increase the amount and diversity of data to train the models and saves the new images in predefined folders.
The script performs the following augmentations:
- RandomRotate90
- Rotate
- GridDistortion
- ShiftScaleRotate
- RandomBrightnessContrast
- Crop

If other augmentation techniques are desired, the script can be changed accordingly. Check [Demo](https://albumentations-demo.herokuapp.com) to test the diffrent augmentation approaches.

## Binary Model
#### Preprocess
[data_preprocess_binary.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/data_preprocess_binary.py): This scripts loads the data from a predefined path, resizes and normalizes the images and transform them into the needed format and creates train, validation and test-sets.
Per default, the script uses the following

- Train/Val/Test-Split = 80% / 10% / 10%
- Height = 256
- Width = 256
- BATCH = 4

These value can be changed directly in the script.
#### Train
[train_binary.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/train_binary.py): Run script to train the model on the prepared datasets. The best result is than saved in .h5 format.
#### Predict
[predict_binary.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/predict_binary.py): Run script to load the pretrained model, and make prediction on unseen data (test-set).
The predicted images are then saved in a predefined folder
#### Visualise
[visualisation_binary.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/visualisation_binary.py): This script includes all the functions to visualise the performance and create plots to show and compare the predicted mask with the ground truth.


## Multiclass Model
#### Preprocess
[data_preprocess_multiclass.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/data_preprocessing_multiclass.py): This scripts loads the data from a predefined path, resizes and normalizes the images and transform them into the needed format and creates train, validation and test-sets.
Per default, the script uses the following

- Train/Val/Test-Split = 80% / 10% / 10%
- Height = 256
- Width = 256
- BATCH = 4

#### Train
[train_multiclass.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/train_multiclass.py): Run script to train the model on the prepared datasets. The best result is than saved in .h5 format.
#### Predict
[predict_multiclass.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/predict_multiclass.py): Run script to load the pretrained model, and make prediction on unseen data (test-set).
#### Visualise
[visualisation_multiclass.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/visualisation_multiclass.py): This script includes all the functions to visualise the performance and create plots to show and compare the predicted mask with the ground truth.


## APP
[helper.py](https://gitlab.propulsion-home.ch/datascience/bootcamp/final-projects/ds-2020-09/caressoma/-/blob/master/library/helper.py): Script includes all functions needed for the app.


