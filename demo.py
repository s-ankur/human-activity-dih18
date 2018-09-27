import sys

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
        text = category + ' ' + "%.2f" % box.get_score()
        cv2.putText(image,
                    text,
                    (xmin, ymin - 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1e-3 * image_h,
                    (160, 200, 80), 2)
        if category == 'person':
            roi = image[ymin:ymax, xmin:xmax, :]
            roi = im2gray(roi)
            roi = cv2.resize(roi, cnn_model.SIZE)
            roi = roi.reshape(1, *roi.shape, 1)

            prediction = phase_two.predict(roi)[0]
            index = np.argmax(prediction)
            activity = cnn_model.categories[index]
            text = activity
            cv2.putText(image,
                        text,
                        (xmin, ymin - 7),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1e-3 * image_h,
                        (0, 200, 80), 2)
    return image


if __name__ == '__main__':
    SHOW = False
    if len(sys.argv) == 2:
        video_path = sys.argv[1]
    else:
        video_path = 0

    phase_one = model_yolo.load_model()
    phase_two = cnn_model.load_model()
    video = Video(video_path)
    dummy_array = np.zeros((1, 1, 1, 1, model_yolo.TRUE_BOX_BUFFER, 4))
    fourcc = cv2.VideoWriter_fourcc(*"MPEG")
    clip = cv2.VideoWriter('demo.avi', fourcc, 30, (416, 416))
    try:
        for image in video:
            inp = cv2.resize(image, (416, 416))
            input_image = inp / 255.
            input_image = input_image[:, :, ::-1]
            input_image = np.expand_dims(input_image, 0)
            netout = phase_one.predict([input_image, dummy_array])
            boxes = decode_netout(netout[0],
                                  obj_threshold=model_yolo.OBJ_THRESHOLD,
                                  nms_threshold=model_yolo.NMS_THRESHOLD,
                                  anchors=model_yolo.ANCHORS,
                                  nb_class=model_yolo.CLASS)

            inp = draw_boxes(inp, boxes)
            clip.write(inp.astype('uint8'))
            if SHOW:
                cv2.imshow('window', image)
                cv2.waitKey(1)
            # print(len(boxes))
    except KeyboardInterrupt:
        pass
    finally:
        clip.release()
        if SHOW:
            destroy_window('window')
