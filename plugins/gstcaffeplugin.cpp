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

#include "gstcaffeplugin.h"
#include "gst/gstdetectionsmeta.h"
#include <fstream>
#include <iostream>
#include <npp.h>
#include <ostream>
#include <sstream>
#include <string.h>
#include <string>
#include <sys/time.h>
GST_DEBUG_CATEGORY_STATIC (gst_caffeplugin_debug);
#define GST_CAT_DEFAULT gst_caffeplugin_debug

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_UNIQUE_ID,
  PROP_NETWORK_TYPE,
  PROP_MODEL_PATH,
  PROP_WEIGHTS_FILENAME,
  PROP_NMS_THRESHOLD,
  PROP_CONFIDENCE_THRESHOLD
};

/* Default values for properties */
#define DEFAULT_UNIQUE_ID 15
#define DEFAULT_EMPTY_STRING ""
#define DEFAULT_NMS_THRE 0.4
#define DEFAULT_CONFIDENCE_THRE 0.5

#define CHECK_CUDA_STATUS(cuda_status,error_str) do { \
  if ((cuda_status) != cudaSuccess) { \
    g_error ("Error: %s in %s at line %d (%s)\n", \
        error_str, __FILE__, __LINE__, cudaGetErrorName(cuda_status)); \
    goto error; \
  } \
} while (0)

static GstStaticPadTemplate gst_caffeplugin_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink", GST_PAD_SINK, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw,format=BGR"));

static GstStaticPadTemplate gst_caffeplugin_src_template =
GST_STATIC_PAD_TEMPLATE ("src", GST_PAD_SRC, GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-raw,format=BGR"));

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define gst_caffeplugin_parent_class parent_class
G_DEFINE_TYPE (GstCaffePlugin, gst_caffeplugin, GST_TYPE_BASE_TRANSFORM);

static void gst_caffeplugin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_caffeplugin_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

static gboolean gst_caffeplugin_set_caps (GstBaseTransform * btrans,
    GstCaps * incaps, GstCaps * outcaps);
static gboolean gst_caffeplugin_start (GstBaseTransform * btrans);
static gboolean gst_caffeplugin_stop (GstBaseTransform * btrans);

static GstFlowReturn gst_caffeplugin_transform_ip (GstBaseTransform * btrans,
    GstBuffer * inbuf);


/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
gst_caffeplugin_class_init (GstCaffePluginClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *gstbasetransform_class;

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;
  gstbasetransform_class = (GstBaseTransformClass *) klass;

  /* Overide base class functions */
  gobject_class->set_property = GST_DEBUG_FUNCPTR (gst_caffeplugin_set_property);
  gobject_class->get_property = GST_DEBUG_FUNCPTR (gst_caffeplugin_get_property);

  gstbasetransform_class->set_caps =
      GST_DEBUG_FUNCPTR (gst_caffeplugin_set_caps);
  gstbasetransform_class->start = GST_DEBUG_FUNCPTR (gst_caffeplugin_start);
  gstbasetransform_class->stop = GST_DEBUG_FUNCPTR (gst_caffeplugin_stop);

  gstbasetransform_class->transform_ip =
      GST_DEBUG_FUNCPTR (gst_caffeplugin_transform_ip);

  /* Install properties */
  g_object_class_install_property (gobject_class, PROP_UNIQUE_ID,
      g_param_spec_uint ("unique-id", "Unique ID",
          "Unique ID for the element. Can be used to identify output of the"
          " element",
          0, G_MAXUINT, DEFAULT_UNIQUE_ID,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NETWORK_TYPE,
      g_param_spec_string ("network", "Network Type",
          "FRCNN or SSD",
          DEFAULT_EMPTY_STRING,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_MODEL_PATH,
      g_param_spec_string ("model-path", "Serialized Engine File",
          "Directory of model data",
          DEFAULT_EMPTY_STRING,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_WEIGHTS_FILENAME,
      g_param_spec_string ("weights-file", "Weights File",
          "Caffe model to use",
          DEFAULT_EMPTY_STRING,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NMS_THRESHOLD,
      g_param_spec_float ("nms", "NMS Threshold",
          "NMS threshold value to use",
          0, 1.0, DEFAULT_NMS_THRE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CONFIDENCE_THRESHOLD,
      g_param_spec_float ("confidence", "Confidence Threshold",
          "Minimum allowable confidence to pass",
          0, 1.0, DEFAULT_CONFIDENCE_THRE,
          (GParamFlags) (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));


  /* Set sink and src pad capabilities */
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_caffeplugin_src_template));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&gst_caffeplugin_sink_template));

  /* Set metadata describing the element */
  gst_element_class_set_details_simple (gstelement_class, "CaffeTRT", "CaffeTRT",
      "Process a 3rdparty example algorithm on objects / full frame", "jayveeangeles");
}

static void
gst_caffeplugin_init (GstCaffePlugin * caffeplugin)
{
  GstBaseTransform *btrans = GST_BASE_TRANSFORM (caffeplugin);

  /* We will not be generating a new buffer. Just adding / updating
   * metadata. */
  gst_base_transform_set_in_place (GST_BASE_TRANSFORM (btrans), TRUE);
  /* We do not want to change the input caps. Set to passthrough. transform_ip
   * is still called. */
  gst_base_transform_set_passthrough (GST_BASE_TRANSFORM (btrans), TRUE);

  /* Initialize all property variables to default values */
  caffeplugin->unique_id = DEFAULT_UNIQUE_ID;
  caffeplugin->network = g_strdup (DEFAULT_EMPTY_STRING);
  caffeplugin->model_path = g_strdup (DEFAULT_EMPTY_STRING);
  caffeplugin->weights_file = g_strdup (DEFAULT_EMPTY_STRING);
  caffeplugin->nms = DEFAULT_NMS_THRE;
  caffeplugin->confidence = DEFAULT_CONFIDENCE_THRE;
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
gst_caffeplugin_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstCaffePlugin *caffeplugin = GST_CAFFEPLUGIN (object);
  switch (prop_id) {
    case PROP_UNIQUE_ID:
      caffeplugin->unique_id = g_value_get_uint (value);
      break;
    case PROP_NETWORK_TYPE:
      if (g_value_get_string (value)) {
        g_free (caffeplugin->network);
        caffeplugin->network = g_value_dup_string (value);
      }
      break;
    case PROP_MODEL_PATH:
      if (g_value_get_string (value)) {
        g_free (caffeplugin->model_path);
        caffeplugin->model_path = g_value_dup_string (value);
      }
      break;
    case PROP_WEIGHTS_FILENAME:
      if (g_value_get_string (value)) {
        g_free (caffeplugin->weights_file);
        caffeplugin->weights_file = g_value_dup_string (value);
      }
      break;
    case PROP_NMS_THRESHOLD:
      caffeplugin->nms = g_value_get_float (value);
      break;
    case PROP_CONFIDENCE_THRESHOLD:
      caffeplugin->confidence = g_value_get_float (value);
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
gst_caffeplugin_get_property (GObject * object, guint prop_id, GValue * value,
    GParamSpec * pspec)
{
  GstCaffePlugin *caffeplugin = GST_CAFFEPLUGIN (object);

  switch (prop_id) {
    case PROP_UNIQUE_ID:
      g_value_set_uint (value, caffeplugin->unique_id);
      break;
    case PROP_NETWORK_TYPE:
      g_value_set_string (value, caffeplugin->network);
      break;
    case PROP_MODEL_PATH:
      g_value_set_string (value, caffeplugin->model_path);
      break;
    case PROP_WEIGHTS_FILENAME:
      g_value_set_string (value, caffeplugin->weights_file);
      break;
    case PROP_NMS_THRESHOLD:
      g_value_set_float (value, caffeplugin->nms);
      break;
    case PROP_CONFIDENCE_THRESHOLD:
      g_value_set_float (value, caffeplugin->confidence);
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
gst_caffeplugin_start (GstBaseTransform * btrans)
{
  GstCaffePlugin *caffeplugin = GST_CAFFEPLUGIN (btrans);
  CaffePluginInitParams init_params = { 
    caffeplugin->network, 
    caffeplugin->model_path,
    caffeplugin->weights_file,
    caffeplugin->nms, 
    caffeplugin->confidence
  };

  if ((!caffeplugin->network)
      || (strlen (caffeplugin->network) == 0)) {
    g_error ("ERROR: network type not set \n");
    goto error;
  }

  if ((!caffeplugin->model_path)
      || (strlen (caffeplugin->model_path) == 0)) {
    g_error ("ERROR: model path not set \n");
    goto error;
  }

  if ((!caffeplugin->weights_file)
      || (strlen (caffeplugin->weights_file) == 0)) {
    g_error ("ERROR: weights file not set \n");
    goto error;
  }

  /* Algorithm specific initializations and resource allocation. */
  caffeplugin->caffepluginlib_ctx = CaffePluginCtxInit (&init_params);

  g_assert (caffeplugin->caffepluginlib_ctx
      && "Unable to create caffe plugin lib ctx \n ");
  GST_DEBUG_OBJECT (caffeplugin, "ctx lib %p \n", caffeplugin->caffepluginlib_ctx);

  // CHECK_CUDA_STATUS (cudaSetDevice (caffeplugin->gpu_id),
  //     "Unable to set cuda device");

  return TRUE;
error:
  if (caffeplugin->caffepluginlib_ctx)
    CaffePluginCtxDeinit (caffeplugin->caffepluginlib_ctx);
  return FALSE;
}

/**
 * Stop the output thread and free up all the resources
 */
static gboolean
gst_caffeplugin_stop (GstBaseTransform * btrans)
{
  GstCaffePlugin *caffeplugin = GST_CAFFEPLUGIN (btrans);

  g_info("stopping caffe\n");
  CaffePluginCtxDeinit (caffeplugin->caffepluginlib_ctx);
  GST_DEBUG_OBJECT (caffeplugin, "ctx lib released \n");

  return TRUE;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
gst_caffeplugin_set_caps (GstBaseTransform * btrans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstCaffePlugin *caffeplugin = GST_CAFFEPLUGIN (btrans);

  /* Save the input video information, since this will be required later. */
  gst_video_info_from_caps (&caffeplugin->video_info, incaps);

  // CHECK_CUDA_STATUS (cudaSetDevice (caffeplugin->gpu_id),
  //     "Unable to set cuda device");

  return TRUE;

error:
  return FALSE;
}

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
gst_caffeplugin_transform_ip (GstBaseTransform * btrans, GstBuffer * inbuf)
{
  GstCaffePlugin *caffeplugin = GST_CAFFEPLUGIN (btrans);
  GstMapInfo in_map_info;
  GstFlowReturn flow_ret = GST_FLOW_OK;

  caffeplugin->frame_num++;
  // CHECK_CUDA_STATUS (cudaSetDevice (caffeplugin->gpu_id),
  //     "Unable to set cuda device");

  memset (&in_map_info, 0, sizeof (in_map_info));
  if (!gst_buffer_map (inbuf, &in_map_info, GST_MAP_READ)) {
    g_error ("Error: Failed to map gst buffer\n");
    return GST_FLOW_ERROR;
  }

  g_assert (in_map_info.size == (
    caffeplugin->video_info.height *
    caffeplugin->video_info.width * 3
  )
      && "Unable to create caffe plugin lib ctx \n ");

  GST_DEBUG_OBJECT (caffeplugin,
      "Processing Frame %" G_GUINT64_FORMAT "\n",
      caffeplugin->frame_num);
      
  cv::Mat img (
    caffeplugin->video_info.height, 
    caffeplugin->video_info.width, CV_8UC3, in_map_info.data
  );

  // hack for now
  caffeplugin->caffepluginlib_ctx->images[0] = img;
  try {
    caffeplugin->caffepluginlib_ctx->results = \
      caffeplugin->caffepluginlib_ctx->inferenceNetwork->infer(caffeplugin->caffepluginlib_ctx->images);
  } catch  (const trt::CaffeRuntimeException& e) {
    g_warning("Exception encountered: %s\n", e.what());
  }

  GstDetectionMetas *metas = GST_DETECTIONMETAS_ADD(inbuf);
  metas->detections_count = 0;
  
  for (auto result: caffeplugin->caffepluginlib_ctx->results[0]) {
    GstObjectDetectionMeta *meta = &metas->detections[metas->detections_count++];

    meta->confidence = result.confidence;
    meta->label = g_strdup(result.label.c_str());
    meta->xmin = static_cast<guint>(result.box.x1);
    meta->ymin = static_cast<guint>(result.box.y1);
    meta->xmax = static_cast<guint>(result.box.x2);
    meta->ymax = static_cast<guint>(result.box.y2);
  }

  gst_buffer_unmap (inbuf, &in_map_info);
  return flow_ret;
}

/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
caffeplugin_plugin_init (GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT (gst_caffeplugin_debug, "caffe", 0, "caffe plugin");

  return gst_element_register (plugin, "caffetrt", GST_RANK_PRIMARY,
      GST_TYPE_CAFFEPLUGIN);
}

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR, GST_VERSION_MINOR, caffeplugin,
    DESCRIPTION, caffeplugin_plugin_init, VERSION, LICENSE, BINARY_PACKAGE, URL)
