/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/

#include "gstyoloplugin.h"
#include "gst/gstdetectionsmeta.h"
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string.h>
#include <string>
#include <sys/time.h>
GST_DEBUG_CATEGORY_STATIC (gst_yoloplugin_debug);
#define GST_CAT_DEFAULT gst_yoloplugin_debug

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_PROCESSING_WIDTH,
  PROP_PROCESSING_HEIGHT,
  PROP_GPU_DEVICE_ID,
  PROP_CONFIG_FILE_PATH
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_PROCESSING_WIDTH 640
#define DEFAULT_PROCESSING_HEIGHT 480
#define DEFAULT_GPU_ID 0
#define DEFAULT_CONFIG_FILE_PATH ""

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_print ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

/* By default NVIDIA Hardware allocated memory flows through the pipeline. We
 * will be processing on this type of memory only. */
static GstStaticPadTemplate gst_yoloplugin_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw,format=BGR"));

static GstStaticPadTemplate gst_yoloplugin_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw,format=BGR"));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_yoloplugin_parent_class parent_class
G_DEFINE_TYPE (GstYoloPlugin, gst_yoloplugin, GST_TYPE_BASE_TRANSFORM);

static void gst_yoloplugin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_yoloplugin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_yoloplugin_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_yoloplugin_start (GstBaseTransform * btrans);
static gboolean gst_yoloplugin_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_yoloplugin_transform_ip (GstBaseTransform * btrans,
    GstBuffer * inbuf);

static void attach_metadata_to_frame (GstDetectionMetas* metas, 
  YoloPluginOutput * output);

static void draw_predictions(cv::Mat &img, GstDetectionMetas* metas);

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_yoloplugin_class_init (GstYoloPluginClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_yoloplugin_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_yoloplugin_get_property);

  gstbasetransform_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_yoloplugin_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_yoloplugin_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_yoloplugin_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_yoloplugin_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element",
          0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_WIDTH,
      g_param_spec_int ("processing-width", "Processing Width",
          "Width of the input buffer to algorithm", 1, G_MAXINT,
          DEFAULT_PROCESSING_WIDTH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PROCESSING_HEIGHT,
      g_param_spec_int ("processing-height", "Processing Height",
          "Height of the input buffer to algorithm", 1, G_MAXINT,
          DEFAULT_PROCESSING_HEIGHT,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID", "Set GPU Device ID", 0,
          G_MAXUINT, 0,
          GParamFlags (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
              GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CONFIG_FILE_PATH,
      g_param_spec_string ("config-file-path", "Plugin config file path",
          "Set plugin config file path",
          DEFAULT_CONFIG_FILE_PATH,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_yoloplugin_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_yoloplugin_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class, "NvYolo", "NvYolo",
      "Process a 3rdparty example algorithm on objects / full frame", "Nvidia");
}

static void
gst_yoloplugin_init (GstYoloPlugin * yoloplugin)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (yoloplugin);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  yoloplugin->unique_id = DEFAULT_UNIQUE_ID;
  yoloplugin->processing_width = DEFAULT_PROCESSING_WIDTH;
  yoloplugin->processing_height = DEFAULT_PROCESSING_HEIGHT;
  yoloplugin->gpu_id = DEFAULT_GPU_ID;
  yoloplugin->config_file_path = g_strdup (DEFAULT_CONFIG_FILE_PATH);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_yoloplugin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      yoloplugin->unique_id = g_value_get_uint (value);
      break;
    case PROP_PROCESSING_WIDTH:
      yoloplugin->processing_width = g_value_get_int (value);
      break;
    case PROP_PROCESSING_HEIGHT:
      yoloplugin->processing_height = g_value_get_int (value);
      break;
    case PROP_GPU_DEVICE_ID:
      yoloplugin->gpu_id = g_value_get_uint (value);
      break;
    case PROP_CONFIG_FILE_PATH:
      if (g_value_get_string (value)) {
        g_free (yoloplugin->config_file_path);
        yoloplugin->config_file_path = g_value_dup_string (value);
      }
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
gst_yoloplugin_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, yoloplugin->unique_id);
      break;
    case PROP_PROCESSING_WIDTH:
      g_value_set_int (value, yoloplugin->processing_width);
      break;
    case PROP_PROCESSING_HEIGHT:
      g_value_set_int (value, yoloplugin->processing_height);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, yoloplugin->gpu_id);
      break;
    case PROP_CONFIG_FILE_PATH:
      g_value_set_string (value, yoloplugin->config_file_path);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/**
 * Initialize all resources and start the output thread
 */
static gboolean
gst_yoloplugin_start (GstBaseTransform * btrans)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (btrans);
  YoloPluginInitParams init_params =
      { yoloplugin->processing_width, yoloplugin->processing_height, yoloplugin->config_file_path };

  GstQuery *queryparams = NULL;
  guint batch_size = 1;

  if ((!yoloplugin->config_file_path)
      || (strlen (yoloplugin->config_file_path) == 0)) {
    g_print ("ERROR: Yolo plugin config file path not set \n");
    goto error;
  }

  yoloplugin->batch_size = 1;

  GST_DEBUG_OBJECT (yoloplugin, "Setting batch-size %d \n",
      yoloplugin->batch_size);

  /* Algorithm specific initializations and resource allocation. */
  yoloplugin->yolopluginlib_ctx =
      YoloPluginCtxInit (&init_params, yoloplugin->batch_size);

  g_assert (yoloplugin->yolopluginlib_ctx
      && "Unable to create yolo plugin lib ctx \n ");
  GST_DEBUG_OBJECT (yoloplugin, "ctx lib %p \n", yoloplugin->yolopluginlib_ctx);
  CHECK_CUDA_STATUS (cudaSetDevice (yoloplugin->gpu_id),
      "Unable to set cuda device");

  yoloplugin->cvmats =
      std::vector < cv::Mat * >(yoloplugin->batch_size, nullptr);
  for (uint k = 0; k < batch_size; ++k) {
    yoloplugin->cvmats.at (k) =
      new cv::Mat (cv::Size(yoloplugin->video_info.width, 
        yoloplugin->video_info.height), CV_8UC3);
        // new cv::Mat (cv::Size (yoloplugin->processing_width,
        //     yoloplugin->processing_height), CV_8UC3);
    if (!yoloplugin->cvmats.at (k))
      goto error;
  }
  GST_DEBUG_OBJECT (yoloplugin, "created CV Mat\n");
  return TRUE;
error:
  if (yoloplugin->yolopluginlib_ctx)
    YoloPluginCtxDeinit (yoloplugin->yolopluginlib_ctx);
  return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_yoloplugin_stop (GstBaseTransform * btrans)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (btrans);

  for (uint i = 0; i < yoloplugin->batch_size; ++i) {
    delete yoloplugin->cvmats.at (i);
  }
  GST_DEBUG_OBJECT (yoloplugin, "deleted CV Mat \n");
  // Deinit the algorithm library
  YoloPluginCtxDeinit (yoloplugin->yolopluginlib_ctx);
  GST_DEBUG_OBJECT (yoloplugin, "ctx lib released \n");
  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_yoloplugin_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (btrans);

  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&yoloplugin->video_info, incaps);

  CHECK_CUDA_STATUS (cudaSetDevice (yoloplugin->gpu_id),
      "Unable to set cuda device");

  return TRUE;

error:
  return FALSE;
}

static void
attach_metadata_to_frame (
  GstDetectionMetas* metas, YoloPluginOutput* output) {

  for (uint i = 0; i < output->numObjects; i++) {
    GstObjectDetectionMeta *meta = \
      &metas->detections[metas->detections_count++];
    
    const auto xmin = output->object[i].left > 0 ? output->object[i].left : 0;
    const auto ymin = output->object[i].top > 0 ? output->object[i].top : 0;

    const auto xmax = xmin + output->object[i].width;
    const auto ymax = ymin + output->object[i].height;
    const auto label = output->object[i].label;

    strcpy(meta->label, label);

    meta->xmin = xmin;
    meta->ymin = ymin;
    meta->xmax = xmax;
    meta->ymax = ymax;
  }
}

static void
draw_predictions(cv::Mat &img, GstDetectionMetas* metas) {
  for (uint i = 0; i < metas->detections_count; i++) {
    GstObjectDetectionMeta *meta = &metas->detections[i];

    cv::rectangle(img, cv::Point(meta->xmin, meta->ymin), \
      cv::Point(meta->xmax, meta->ymax), cv::Scalar(175, 107, 75), 3);
    cv::putText(img, meta->label, cv::Point(meta->xmin, meta->ymin), \
      cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
  }
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_yoloplugin_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstYoloPlugin *yoloplugin = GST_YOLOPLUGIN (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_OK;
  gdouble scale_ratio;
  std::vector < YoloPluginOutput * >outputs (yoloplugin->batch_size, nullptr);

  GstDetectionMetas *metas = GST_DETECTIONMETAS_ADD(inbuf);
  metas->detections_count = 0;

  guint batch_size = yoloplugin->batch_size;

  cv::Mat img (
    yoloplugin->video_info.height, 
    yoloplugin->video_info.width, CV_8UC3);

  yoloplugin->frame_num++;
  // CHECK_CUDA_STATUS (cudaSetDevice (yoloplugin->gpu_id),
  //     "Unable to set cuda device");


  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_print ("Error: Failed to map gst buffer\n");
    flow_ret = GST_FLOW_ERROR;
    goto error;
  }

  g_assert (in_map_info.size == (
    yoloplugin->video_info.height *
    yoloplugin->video_info.width * 3
  )
      && "Unable to create caffe plugin lib ctx  ");

  GST_DEBUG_OBJECT (yoloplugin,
      "Processing Frame %" G_GUINT64_FORMAT "n", yoloplugin->frame_num);

  img.data = in_map_info.data;

  // Hack since batch size is always one
  *(yoloplugin->cvmats.at(0)) = img;

  // Process to get the outputs
  outputs =
      YoloPluginProcess (yoloplugin->yolopluginlib_ctx, yoloplugin->cvmats);

  for (uint k = 0; k < outputs.size (); ++k) {
    if (!outputs.at (k))
      continue;

    attach_metadata_to_frame(metas, outputs.at(k));

    if (yoloplugin->yolopluginlib_ctx->inferParams.printPredictionInfo) {
      draw_predictions(img, metas);
    }

    free (outputs.at (k));
  }

  flow_ret = GST_FLOW_OK;

error:
  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
yoloplugin_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_yoloplugin_debug, "yolo", 0, "yolo plugin");

  return gst_element_register (plugin, "yololib", GST_RANK_PRIMARY,
      GST_TYPE_YOLOPLUGIN);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, yoloplugin,
    DESCRIPTION, yoloplugin_plugin_init, VERSION, LICENSE, BINARY_PACKAGE, URL)
