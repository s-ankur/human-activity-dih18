from model_yolo import *
from utility.cv_utils import *


if len(sys.argv) == 2:
    video_path = sys.argv[1]
else:
    video_path = 0
video = Video(video_path)
dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
fourcc = cv2.VideoWriter_fourcc(*"MPEG")
clip = cv2.VideoWriter('demo.avi', fourcc, 5, (416, 416), True)


try :
    for image in video:
        input_image = cv2.resize(image, (416, 416))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        netout = model.predict([input_image, dummy_array])
        boxes = decode_netout(netout[0], 
                              obj_threshold=OBJ_THRESHOLD,
                              nms_threshold=NMS_THRESHOLD,
                              anchors=ANCHORS, 
                              nb_class=CLASS)
        image = draw_boxes(image, boxes, labels=LABELS)
        clip.write(image)
        cv2.imshow('window', frame)
        cv2.waitKey(1)
        print(len(boxes))
except KeyboardInterrupt:
    pass
finally:
    clip.release()
    destroy_window('window')
