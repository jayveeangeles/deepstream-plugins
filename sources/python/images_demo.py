#!/usr/bin/python3
import sys
import cv2
import time
import logging
from pyyolo import PyYoloV3

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Yolo Inference')
    parser.add_argument("-f", "--flagfile", type=str, help='config', default="yolov3.txt")

    args = vars(parser.parse_args())

    logging.basicConfig(format='%(asctime)s := [%(levelname)-8s] %(message)s', level=logging.DEBUG)

    # please download pictures from pjreddie darknet, under data folder
    logging.info("Loading Image")
    imgKite = cv2.imread('/tmp/kite.jpg')
    imgDog = cv2.imread('/tmp/dog.jpg')
    imgHorses = cv2.imread('/tmp/horses.jpg')

    logging.info("Starting Inference")
    YoloV3 = PyYoloV3(sys.argv)

    results = YoloV3.detect(imgKite, bytes("imgKite", encoding="utf-8"))
    logging.info("---Kite---")
    for result in results:
        logging.info("class %s (%f)" % (result["label"], result["prob"]))

    results = YoloV3.detect(imgDog, bytes("imgDog", encoding="utf-8"))
    logging.info("---Dog---")
    for result in results:
        logging.info("class %s (%f)" % (result["label"], result["prob"]))

    results = YoloV3.detect(imgHorses, bytes("imgHorses", encoding="utf-8"))
    logging.info("---Horses---")
    for result in results:
        logging.info("class %s (%f)" % (result["label"], result["prob"]))