from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import config
from utility.cv_utils import *

categories = ['shake_hands', 'hugging', 'kicking', 'standing', 'punching', 'pushing']

"""
# location to preprocessed data
loc_src_1 = "./data/img_preprocessed_set_1/"
loc_src_2 = "./data/img_preprocessed_set_2/"


"""

DATA_PATH = 'sdha2010'
VIDEO_PATH = 'videos'
IMAGE_PATH = 'images'


def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Size of original images
    # max-min range --> (332-216, 612, 244)
    # avg range --> (260, 380)
    img_gray = cv2.resize(img_gray, (70, 95))
    img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_norm = np.empty_like(img_gauss)
    img_norm = cv2.normalize(img_gauss, img_norm, 0, 255, cv2.NORM_MINMAX)
    return img_norm


def extract():
    os.mkdir(os.path.join('datasets', DATA_PATH, IMAGE_PATH))
    file_list = os.listdir(os.path.join('datasets', DATA_PATH, VIDEO_PATH))
    for label, category in enumerate(categories):
        category_files = filter(lambda file: get_class_label(file) == label, file_list)
        for seq_num, file in enumerate(category_files):
            generate_images(file, seq_num, label)


def generate_images(video_file_name, seq_num, label):
    ffmpeg_cmd = wrapFfmpegCmd(video_file_name, label, seq_num)
    print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)


def get_class_label(video_file_name):
    file_parts = video_file_name.split('_')
    class_of_file = int(file_parts[2].split('.')[0])
    return class_of_file


def get_class_number(video_file_name):
    file_parts = video_file_name.split('_')
    class_of_file = int(file_parts[0])
    return class_of_file


def wrapFfmpegCmd(file_name, cls_num, seq_num, frame_rate=15):
    """Generates ffmpeg command to conver video into images"""
    file_loc = os.path.join('datasets', DATA_PATH, VIDEO_PATH, file_name)
    gen_file_name = "%s_%s_" % (cls_num, seq_num) + "%d.png"
    gen_file_loc = os.path.join('datasets', DATA_PATH, IMAGE_PATH, gen_file_name)
    command = "ffmpeg -i %s -r %s %s" % (file_loc, frame_rate, gen_file_loc)
    return command


def load_data():
    data = []
    labels = []
    file_names = os.listdir(os.path.join('datasets', DATA_PATH, IMAGE_PATH))
    for file in file_names:
        image = imread(os.path.join('datasets', DATA_PATH, IMAGE_PATH, file))
        image = cv2.resize(image, config.SIZE)
        if config.CHANNELS == 1:
            image = im2gray(image).reshape(*image.shape[:-1], 1)
        data.append(image)
        labels.append(get_class_number(file))
    X = np.array(data)
    X = X / 255.
    y = np.array(labels)
    y = np_utils.to_categorical(y, len(categories))

    return train_test_split(X, y, test_size=config.TEST_TRAIN_SPLIT)


def load_data3d():
    pass


URL = r"http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set1.zip"

def download():
    print("Downloading file")
    download_file(URL,'sdha.zip')
    print("Done")
