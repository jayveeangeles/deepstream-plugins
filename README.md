# TensorRT Caffe Gstreamer Plugin

## Prerequisites
1. CUDA Toolkit
2. TensorRT
3. Gstreamer

## To Build
1. ```mkdir build;cd build```
2. ```cmake -DCMAKE_BUILD_TYPE=Release -DTRT_SDK_ROOT=<TensorRT root folder> ..```
3. ```make install```

## Plugin Parameters
1. __**network**__ network type, either FRCNN or Caffe
2. __**model-path**__ location of deployment files
3. __**weights-file**__ filename of caffemodel (just the filename;no paths)
4. __**nms**__ (optional) NMS threshold value; defaults to 0.4
5. __**confidence**__ (optional) min confidence level; defaults to 0.5 

## Sample Gstreamer Run
```GST_DEBUG=3  gst-launch-1.0 v4l2src ! 'video/x-raw,width=1280,height=720' ! videoconvert ! caffetrt network=FRCNN model-path=/workspace/models/deploy weights-file=<model name>.caffemodel confidence=0.7 ! videoconvert ! xvimagesink -e```