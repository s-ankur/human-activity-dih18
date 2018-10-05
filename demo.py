from math import isclose

from imutils.object_detection import non_max_suppression

import model as cnn_model
import model_yolo
from utility.cv_utils import *
from utils import *


def draw_boxes(image, boxes):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        category = model_yolo.categories[box.get_label()]
        text = category + ' ' + "%.2f%%" % (box.get_score() * 100)
        cv2.putText(image,
                    text,
                    (xmin - 2, ymin - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    (255, 255, 255), 2)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (200, 200, 200), 3)

        if category == 'person':
            roi = image[ymin:ymax, xmin:xmax, :]
            roi = im2gray(roi)
            roi = cv2.resize(roi, cnn_model.SIZE)
            roi = roi.reshape(1, *roi.shape, 1)

            prediction = classifier.predict(roi)[0]
            index = np.argmax(prediction)
            activity = cnn_model.categories[index]
            text = activity
            cv2.putText(image,
                        text,
                        (xmin - 2, ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * image_h,
                        (255, 255, 255), 2)
    return image


def preprocess(image):
    # y, x = image.shape[:2]
    # t = min(x, y)
    # image = image[:t, :t, :]
    image = cv2.resize(image, (1024, 1024))
    inp = cv2.resize(image, (416, 416))
    return inp


def rects_from_boxes(boxes, shape):
    rects = np.array(
        [[x * shape[1], y * shape[0], x2 * shape[1], y2 * shape[0]] for (x, y, x2, y2) in boxes])
    return rects


class HOGDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image):
        hogout = self.hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        boxes = decode_hogout(hogout, image)
        return boxes


class YOLODectector:
    def __init__(self):
        self.model = model_yolo.load_model()
        self.dummy_array = np.zeros((1, 1, 1, 1, model_yolo.TRUE_BOX_BUFFER, 4))

    def detect(self, inp):
        #        now =time()
        input_image = inp / 255.
        input_image = input_image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)
        netout = self.model.predict([input_image, self.dummy_array])
        #        then =time()
        boxes = decode_netout(netout[0],
                              obj_threshold=.45,
                              nms_threshold=model_yolo.NMS_THRESHOLD,
                              anchors=model_yolo.ANCHORS,
                              nb_class=model_yolo.CLASS)
        #        print(then-now,time()-then)
        return boxes


def suppress(boxes, shape):
    rects = rects_from_boxes(boxes, shape)
    picked = non_max_suppression(rects, overlapThresh=.065)
    # print(picked)
    ans = []
    for (x, y, x2, y2) in picked:
        for box in boxes:
            (X, Y, X2, Y2) = box
            if isclose(x, X * shape[0], abs_tol=2) and \
                    isclose(y, Y * shape[1], abs_tol=2) and \
                    isclose(x2, X2 * shape[0], abs_tol=2) and \
                    isclose(y2, Y2 * shape[1], abs_tol=2):
                ans.append(box)
    return ans


def find_activity(boxes):
    persons = list(filter(lambda box: box.get_label() == 'person', boxes))
    if len(persons) < 2:
        return boxes
    ymin = float('inf')
    ymax = float('-inf')
    xmin = float('inf')
    xmax = float('-inf')
    s = 0
    for person in persons:
        ymin = max(person.ymin, ymin)
        xmin = max(person.xmin, xmin)
        xmax = min(person.xmax, xmax)
        ymax = min(person.ymax, ymax)
        s += person.get_score()

    activity = BoundBox(xmin, ymin, xmax, ymax)
    activity.score = s / len(persons)
    activity.label = 'activity'
    boxes.append (activity)
    return boxes

URL = 'https://drive.google.com/file/d/1ecI2V5rx1_uZ3cMY6q9yNDujfQo_opn1/view?usp=sharing'

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', nargs='?', default=0)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--hog', action='store_true')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--suppress', action='store_true')

    args = parser.parse_args()

    if args.download:
        download_file(URL, 'weights_coco.h5')

    print('Input Stream:', args.video_path)

    # tracker = ObjectTracker()
    if args.hog:
        detector = HOGDetector()
    else:
        detector = YOLODectector()

    classifier = cnn_model.load_model()
    video = Video(args.video_path)
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    clip = cv2.VideoWriter('demo.avi', fourcc, 30, (1024, 1024))

    for image in video:
        try:
            inp = preprocess(image)
            detected = detector.detect(inp)
            if args.suppress:
                detected = suppress(detected, inp.shape)
            # print(len(detected),len(selected))
            boxes = find_activity(boxes)
            draw_boxes(inp, detected)
            clip.write(inp.astype('uint8'))
            if args.show:
                cv2.imshow('window', inp)
                key = cv2.waitKey(1)
            # print (len(boxes))
        except  KeyboardInterrupt:
            break
        except Exception:
            raise

    clip.release()
    if args.show:
        destroy_window('window')
