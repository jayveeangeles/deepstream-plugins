from build.YoloV3 import YoloV3, InferParams, NetworkInfo
import cv2 as cv
# import line_profiler
import numpy as np

baseDirectory = '/home/user/Documents/yolov3_onnx_plugin'

n = NetworkInfo()
i = InferParams()

n.networkType = "yolov3"
n.configFilePath = "{}/yolov3.cfg".format(baseDirectory)
n.wtsFilePath = "{}/yolov3.weights".format(baseDirectory)
n.labelsFilePath = "{}/coco_labels.txt".format(baseDirectory)
n.precision = "kHALF"
n.deviceType = "kGPU"
n.calibrationTablePath = "/tmp/calibrate.txt"
n.enginePath = "/tmp/yolo.trt"
n.inputBlobName = "data"

i.printPerfInfo = False
i.printPredictionInfo = False
i.calibImages = "/tmp/list.txt"
i.calibImagesPath = "/tmp/path.txt"
i.probThresh = 0.7
i.nmsThresh = 0.5

def draw_bboxes(image_raw, results):
  for result in results:
    cv.rectangle(image_raw, (int(result.box.x1),int(result.box.y1)),(int(result.box.x2),int(result.box.y2)),(255,0,0), 3)
    cv.putText(image_raw, yolov3.getClassName(result.label), (int(result.box.x1), int(result.box.y1)), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

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
  yolov3 = YoloV3(1, n, i)
  cap = cv.VideoCapture(input_video_path)

  main()