#!/usr/bin/python3
import sys
import cv2
import time
import logging
from pyyolo import PyYoloV3

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Yolo Inference')
    parser.add_argument("-f", "--flagfile", type=str, help='config', default="config/yolov3.txt")

    args = vars(parser.parse_args())

    logging.basicConfig(format='%(asctime)s := [%(levelname)-8s] %(message)s', level=logging.DEBUG)

    logging.info("Starting Inference")

    average_time = 0
    cap = cv2.VideoCapture("test.mp4")
    YoloV3 = PyYoloV3([sys.argv[0], ('--flagfile=%s' % args['flagfile'])])

    while True:
            r, frame = cap.read()
            if r:
                start_time = time.time()

                # Only measure the time taken by YOLO and API Call overhead

                results = YoloV3.detect(frame, bytes("video_frame", encoding="utf-8"))
                # logging.info(results)

                end_time = time.time()
                average_time = average_time * 0.8 + (end_time-start_time) * 0.2

                # for logging, specifying --print_perf_info in the config file will enable
                # printing of cython performance logging information
                logging.info("Total Time: %f : %f" % (end_time-start_time, average_time))

                # Data in YOLO34PY Format
                # https://github.com/madhawav/YOLO3-4-Py
                for cat, score, bounds in results:
                    x, y, w, h = bounds
                    cv2.rectangle(frame, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                    cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

                # Data in dictionary format
                # for result in results:
                #     x1 = result["box"]["x1"]
                #     y1 = result["box"]["y1"]
                #     x2 = result["box"]["x2"]
                #     y2 = result["box"]["y2"]
                #     cv2.rectangle(frame, (int(x1),int(y1)),(int(x2),int(y2)),(255,0,0))
                #     cv2.putText(frame, result["label"], (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

                cv2.imshow("preview", frame)


            k = cv2.waitKey(1)
            if k == 0xFF & ord("q"):
                break