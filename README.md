
# human-activity-dih18
[![ForTheBadge made-with-python](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

Human Activity Recognition Using Deep Neural Network

Currently use Neural Network to classify Human Activities
We use 3 different types of models cnn2d cnn3d and lstm to classify human activites.
Till now 2 different datasets are used. UCF101 and SDHA2010

The project depends on keras, sklearn, opencv, tqdm, requests and their respective dependencies<br>
Use `pip install -r requirements.txt` to install dependencies.

To train, do the following 
 1. Edit the config[3d].py file to set the desired hyperparameters and select the `DATASET` as `ucf101` or `sdha2010`.
 3. To download the dataset automatically use the command
    `python dataset.py download`
    
    OR
    
    Download the dataset manually from  http://crcv.ucf.edu/ICCV13-Action-Workshop/download.html
 4. Extract the dataset in a folder named `videos` in the dataset folder.
 5. Run `python dataset.py extract[3d]` for the extraction of frames from the dataset.
 6. Run fit[3d].py to train a model from the extracted images
 7. All results and vmetrics will be stored in the `results` folder on completion.
 
To use 3d networks do the above steps with 3d.py files
 
For actual live video demonstration we use a stacked yolo model to extract the roi for our models.
 

Do checkout
https://github.com/experiencor/keras-yolo2
