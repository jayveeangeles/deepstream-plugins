#!/usr/bin/python3
from __future__ import division, print_function, absolute_import

import os
import sys
import logging
import subprocess
from setuptools import setup, Extension
# from setuptools.extension import Extension
from Cython.Build import cythonize
from distutils.command.clean import clean
from distutils.command.build import build
from Cython.Distutils import build_ext

def create_soft_links():
  logging.info("create soft links for other source files")
  cwd = os.getcwd()
  logging.info("getting current directory %s" % cwd)
  logging.info("getting working directory %s" % working_dir)
  ret = subprocess.Popen(['/bin/bash', 'soft_link_libs.sh', working_dir, cwd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=('%s/lib' % cwd))
  for line in iter(ret.stdout.readline, ''):
    if len(line) != 0:
      logging.info(line.rstrip())
    else:
      break

  for line in iter(ret.stderr.readline, ''):
    if len(line) != 0:
      logging.warning(line.rstrip())
    else:
      break

def setup_yolo_libs():
  logging.info("building YOLO inference libs")
  cwd = os.getcwd()
  logging.info("getting current directory %s" % cwd)
  ret = subprocess.Popen(['make', '-j', '$(nproc)'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, cwd=('%s/lib' % cwd))

  for line in iter(ret.stdout.readline, ''):
    if len(line) != 0:
      logging.info(line.rstrip())
    else:
      break

  for line in iter(ret.stderr.readline, ''):
    if len(line) != 0:
      logging.warning(line.rstrip())
    else:
      break

  if ret.wait() == 0:
      logging.info("YOLO libs successfully built")
      return True
  else:
      logging.critical("failed to build YOLO libs")
      return False

class CustomBuild(build):
  def run(self):
      create_soft_links()
      if setup_yolo_libs():
        build.run(self)

class CustomBuildExt(build_ext):
  def run(self):
      create_soft_links()
      if setup_yolo_libs():
        build_ext.run(self)

package_name = 'pyyolo'
pkg_version = '0.1'

cmd_class = {"build": CustomBuild, "build_ext": CustomBuildExt}

# src files
src_files = ['pyyolo.pyx', 'bridge.cpp']

# compilation arguments
comp_args = ['-O2',
             '-std=c++11',
             '-g',
             '-Wall',
             '-Wunused-function',
             '-Wunused-variable',
             '-Wfatal-errors',
             '-fPIC']

# additional linkin arguments (runtime)
link_args = ['-Wl,-rpath="/usr/lib/aarch64-linux-gnu"',
             '-Wl,-rpath="/usr/local/cuda-10.0/lib64"',
             '-Wl,-rpath="/usr/local/lib"', 
             '-Wl,-rpath="/usr/lib"']

# libraries to use
libs_args = ['nvinfer',
             'nvinfer_plugin',
             'cudart', 
             'cublas', 
             'curand', 
             'opencv_core', 
             'opencv_imgproc', 
             'opencv_imgcodecs', 
             'opencv_highgui', 
             'opencv_dnn', 
             'gflags', 
             'stdc++fs']

# library directories
libs_dirs = ["/usr/lib/aarch64-linux-gnu",
             "/usr/local/cuda-10.0/lib64",
             "/usr/local/lib",
             "/usr/lib/aarch64-linux-gnu/",
             "/usr/lib"]

# include directories
incl_dirs = ['/usr/local/include',
            '/usr/local/cuda-10.0/include',
            '/usr/include/aarch64-linux-gnu',
            '/usr/include',
            ('%s/lib' % os.getcwd())]

# Yolo Inference Libs
extra_obj = ['./lib/build/calibrator.o',
             './lib/build/ds_image.o',
             './lib/build/kernels.o',
             './lib/build/trt_utils.o', 
             './lib/build/yolo.o', 
             './lib/build/yolo_config_parser.o',
             './lib/build/yolov3.o', 
             './lib/build/plugin_factory.o']

if __name__ == "__main__":
  logging.basicConfig(format='%(asctime)s := [%(levelname)-8s] %(message)s', level=logging.DEBUG)
  
  logging.info("using g++ as CC and CXX compiler")
  os.environ['CC'] = 'g++'
  os.environ['CXX'] = 'g++'

  try:
    working_dir = os.environ['SRCDIR']
  except:
      logging.critical("please specify correct working path")
      logging.critical("e.g. SRCDIR=$(pwd) pip3 install .")
      sys.exit(-1)
  
  if 'YOLO34PY_FORMAT' in os.environ and os.environ['YOLO34PY_FORMAT']:
    use_yolo34py_format = True
    logging.info("using yolo34py format")
  else: use_yolo34py_format = False

  logging.info('current working directory %s' % working_dir)
  logging.info('current current directory %s' % os.getcwd())
  
  cython_ext = Extension(
      package_name,
      sources=src_files,
      include_dirs=incl_dirs,
      library_dirs=libs_dirs,
      libraries=libs_args,
      extra_link_args=link_args,
      extra_compile_args=comp_args,
      extra_objects=extra_obj,
      language='c++',
  )

  setup(
    cmdclass=cmd_class,
    name=package_name,
    version=pkg_version,
    description = "Python3 binding for TensorRT based YOLOv3 Inference",
    author = "John Vincent Angeles",
    author_email = "jayveeangeles@gmail.com",
    url = "https://github.com/jayveeangeles/deepstream-plugins",
    platforms = ["Linux L4T"],
    setup_requires = ["cython", "numpy"],
    install_requires = ["numpy"],
    provides = [package_name],
    ext_modules=cythonize([cython_ext], compile_time_env=dict(YOLO34PY_FORMAT=use_yolo34py_format))
  )
