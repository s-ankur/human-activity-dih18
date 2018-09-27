# human-activity-dih18
Human Activity Recognition Using Deep Neural Network

Currently use Neural Network to classify Human Activities

To run, do the following 
 1. Download UCF101 dataset and unpack it into a folder named  videos .
    http://crcv.ucf.edu/ICCV13-Action-Workshop/download.html
 2. Remove any class folder you wish to disregard.
 3. Edit the config[3d].py file to set the desired hyperparameters.
 4. Run extract[3d].py to extract random frames from the videos.
 5. Run fit[3d].py to make a model from the images
 6. View online prediction using predict[3d].py <filename> . Leave blank to use webcam

To use 3d networks do the above steps with 3d.py files

Do checkout
https://github.com/experiencor/keras-yolo2
