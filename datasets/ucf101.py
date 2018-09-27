"""
UCF101 Dataset
"""

import random
import sys

from keras.utils import generic_utils, np_utils
from sklearn.model_selection import train_test_split

import config
import config3d
from utility.cv_utils import *

DATA_PATH = 'ucf101'
VIDEO_PATH = 'videos'
IMAGE_PATH = 'images'
CLIP_PATH = 'clips'

categories = ['ApplyEyeMakeup', 'ApplyLipstick', 'Archery', 'BabyCrawling', 'BalanceBeam', 'BandMarching',
              'BaseballPitch', 'Basketball', 'BasketballDunk',
              'BenchPress', 'Biking', 'Billiards', 'BlowDryHair', 'BlowingCandles', 'BodyWeightSquats', 'Bowling',
              'BoxingPunchingBag', 'BoxingSpeedBag',
              'BreastStroke', 'BrushingTeeth', 'CleanAndJerk', 'CliffDiving', 'CricketBowling', 'CricketShot',
              'CuttingInKitchen', 'Diving', 'Drumming',
              'Fencing', 'FieldHockeyPenalty', 'FloorGymnastics', 'FrisbeeCatch', 'FrontCrawl', 'GolfSwing', 'Haircut',
              'Hammering', 'HammerThrow', 'HandstandPushups', 'HandstandWalking', 'HeadMassage', 'HighJump',
              'HorseRace', 'HorseRiding', 'HulaHoop', 'IceDancing', 'JavelinThrow', 'JugglingBalls', 'JumpingJack',
              'JumpRope', 'Kayaking', 'Knitting', 'LongJump', 'Lunges', 'MilitaryParade', 'Mixing', 'MoppingFloor',
              'Nunchucks', 'ParallelBars', 'PizzaTossing', 'PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute',
              'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin', 'PoleVault',
              'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'Rafting', 'RockClimbingIndoor', 'RopeClimbing', 'Rowing',
              'SalsaSpin', 'ShavingBeard', 'Shotput', 'SkateBoarding', 'Skiing', 'Skijet', 'SkyDiving',
              'SoccerJuggling', 'SoccerPenalty', 'StillRings', 'SumoWrestling', 'Surfing', 'Swing', 'TableTennisShot',
              'TaiChi', 'TennisSwing',
              'ThrowDiscus', 'TrampolineJumping', 'Typing', 'UnevenBars', 'VolleyballSpiking', 'WalkingWithDog',
              'WallPushups', 'WritingOnBoard', 'YoYo']

print("Categories Found ", len(categories))
if len(categories) < 2:
    raise ValueError("Cannot classify %d class" % len(categories))


def load_data():
    ret_X = []
    ret_y = []
    for train_or_test in 'train', 'test':
        data = []
        labels = []
        for label, category in enumerate(categories):
            files = glob.glob(os.path.join('datasets', DATA_PATH, IMAGE_PATH, train_or_test, category, '*.jpg'))
            print("%3d. Category %-50s  %-7d files" % (label, category, len(files)))
            for file in files:
                image = imread(file)
                image = cv2.resize(image, config.SIZE)
                if config.CHANNELS == 1:
                    image = im2gray(image).reshape(*image.shape[:-1], 1)
                data.append(image)
                labels.append(label)
        X = np.array(data)
        X = X / 255.
        y = np.array(labels)
        y = np_utils.to_categorical(y, len(categories))

        print('X_%s.shape:' % train_or_test, X.shape)
        print('y_%s.shape:' % train_or_test, y.shape)

        ret_X.append(X)
        ret_y.append(y)
    return ret_X + ret_y


def load_data3d():
    ret_X = []
    ret_y = []
    for train_or_test in 'train', 'test':
        data = []
        labels = []
        for label, category in enumerate(categories):
            files = glob.glob(os.path.join('datasets', DATA_PATH, CLIP_PATH, train_or_test, category, '*.avi'))
            print("%3d. Category %-50s  %-7d files" % (label, category, len(files)))
            for file in files:
                video = Video(file)
                frame_array = []
                for frame in video:
                    if config3d.CHANNELS == 1:
                        frame = im2gray(frame).reshape(*frame.shape[:-1], 1)
                    frame_array.append(frame)
                frame_array = np.array(frame_array)
                data.append(frame_array)
                labels.append(label)
        if not config3d.EXTRACT:
            X = np.array(data).transpose((0, 2, 3, 4, 1))
            X = X.reshape((X.shape[0], *config3d.SIZE3D, config3d.DEPTH, config3d.CHANNELS))
            X = X / 255
        else:
            from .extractor import Extractor
            extractor = Extractor()
            X = []
            for frame_array in data:
                frame_array = extractor.extract(frame_array)
                X.append(frame_array)
            X = np.array(X)
        y = np.array(labels)
        y = np_utils.to_categorical(y, len(categories))

        print('X_%s.shape:' % train_or_test, X.shape)
        print('y_%s.shape:' % train_or_test, y.shape)

        ret_X.append(X)
        ret_y.append(y)
    return ret_X + ret_y


def extract():
    os.mkdir(IMAGE_PATH)
    for label, category in enumerate(categories):
        files = glob.glob(os.path.join('datasets', DATA_PATH, VIDEO_PATH, category, '*'))
        print("%3d. Category %-50s  %-7d files" % (label, category, len(files)))
        train_video_paths, test_video_paths = train_test_split(files, test_size=config.TEST_TRAIN_SPLIT)
        for train_or_test, video_paths in ('train', train_video_paths), ('test', test_video_paths):
            print('Extracting frames from %d %s videos' % (len(video_paths), train_or_test))
            destination_directory = os.path.join('dataset', DATA_PATH, IMAGE_PATH, train_or_test, category)
            os.makedirs(destination_directory, exist_ok=True)
            image_id = 0
            for video_path in video_paths:
                video = Video(video_path)
                progressbar = generic_utils.Progbar(len(video))
                frame_id = 0
                for frame in video:
                    progressbar.add(1)
                    if config.CHANCE > random.random():
                        image_path = os.path.join(destination_directory, str(image_id) + '.jpg')
                        imwrite(image_path, frame)
                        image_id += 1
                        frame_id += 1
                sys.stdout.write('\b' * 100)
                sys.stdout.write('Extracted %d frames from %s' % (frame_id, video_path))
                print()


def extract3d():
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    os.mkdir(CLIP_PATH)
    for label, category in enumerate(categories):
        files = glob.glob(os.path.join('datasets', DATA_PATH, VIDEO_PATH, category, '*'))
        print("%3d. Category %-50s  %-7d files" % (label, category, len(files)))
        train_video_paths, test_video_paths = train_test_split(files, test_size=config.TEST_TRAIN_SPLIT)
        for train_or_test, video_paths in ('train', train_video_paths), ('test', test_video_paths):
            print('Extracting videos from %d %s videos' % (len(video_paths), train_or_test))
            destination_directory = os.path.join('dataset', DATA_PATH, IMAGE_PATH, train_or_test, category)
            os.makedirs(destination_directory, exist_ok=True)
            clip_id = 0
            for video_path in video_paths:
                print(video_path)
                video = Video(video_path)
                while True:
                    ret, frame = video.input_video.read()
                    if not ret:
                        break
                    if config3d.CHANCE > random.random():
                        clip_name = os.path.join(CLIP_PATH, train_or_test, category, str(clip_id) + '.avi')
                        clip = cv2.VideoWriter(clip_name, fourcc, 5, config3d.SIZE3D, True)
                        clip_id += 1
                        delete = False
                        for i in range(config3d.DEPTH + 1):
                            ret, frame = video.input_video.read()
                            if ret:
                                frame = cv2.resize(frame, config3d.SIZE3D)
                                clip.write(frame)
                            else:
                                delete = True
                        clip.release()
                        if delete:
                            os.unlink(clip_name)
                            print("Deleting incomplete clip", clip_name)


def download():
    import urllib.request
    URL = "http://crcv.ucf.edu/data/UCF101/UCF101.rar"
    destination = os.path.join('datasets', DATA_PATH, 'ucf101.rar')
    urllib.request.urlretrieve(URL, destination)
