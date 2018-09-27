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

[A report on the various methods used](https://docs.google.com/document/d/1DgRkhRfk6W0o-6MjYYescqEKPt1YiK5UZsb9LRnsMPg/edit?usp=sharing)

[Final Report](https://docs.google.com/document/d/1m4X8QbcFxY7Qtkzl4Jqy1hoqzm7EKtlWvxqDsUe9ezw/edit?usp=sharing_eil&ts=5babd569)

Do checkout
https://github.com/experiencor/keras-yolo2
