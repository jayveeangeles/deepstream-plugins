from build.YoloV3 import YoloV3
import cv2 as cv
# import line_profiler
import numpy as np

baseDirectory = '/home/user/Documents/yolov3_onnx_plugin'

NetworkInfo = {
  "networkType" : "yolov3",
  "configFilePath" : "{}/yolov3.cfg".format(baseDirectory),
  "wtsFilePath" : "{}/yolov3.weights".format(baseDirectory),
  "labelsFilePath" : "{}/coco_labels.txt".format(baseDirectory),
  "precision" : "kHALF",
  "deviceType" : "kGPU",
  "calibrationTablePath" : "/tmp/calibrate.txt",
  "enginePath" : "/tmp/yolo.engine",
  "inputBlobName" : "data"
}

InferParams = {
  "printPerfInfo" : False,
  "printPredictionInfo" : False,
  "calibImages" : "/tmp/list.txt",
  "calibImagesPath" : "/tmp/path.txt",
  "probThresh" : 0.7,
  "nmsThresh" : 0.5
}

def draw_bboxes(image_raw, results):
  for result in results:
    # cv.rectangle(image_raw, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0), 3)
    # cv.putText(image_raw, all_categories[category], (int(x), int(y)), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

    cv.rectangle(image_raw, (int(result.box.x1),int(result.box.y1)),(int(result.box.x2),int(result.box.y2)),(255,0,0), 3)
    # cv.putText(image_raw, yolov3.getClassName(category), (int(x1), int(y1)), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

  return image_raw
  
input_video_path = '{}/test.mp4'.format(baseDirectory)

# @profile
def main():
  # start of main loop
  while True:
    # read frame
    r, frame = cap.read()
    np_frame = np.array(frame, dtype=np.float32, order='C')

    results = yolov3.detect(np_frame)

    output_frame = draw_bboxes(frame, results)
    cv.imshow('preview', output_frame)
    
    k = cv.waitKey(1)
    if k == 0xFF & ord('q'):
      break

if __name__ == '__main__':
  yolov3 = YoloV3(1, NetworkInfo, InferParams)
  cap = cv.VideoCapture(input_video_path)

  main()