PACKAGE_NAME = pyyolo
MAKE := $(MAKE) --no-print-directory
SHELL = bash
WORKING_DIR := $(shell pwd)

all: install

help:
		@echo "Makefile for $(PACKAGE_NAME)"
		@echo
		@echo 'Usage:'
		@echo
		@echo '    make                  install package thru pip'
		@echo '    make install          install package thru pip'
		@echo '    make install-yolo34py install package thru pip in yolo34py format'
		@echo '    make reset            clean and reinstall'
		@echo '    make reset-yolo34py   clean and reinstall in yolo34py format'
		@echo '    make test             check package can be imported'
		@echo '    make build            build ext in place'
		@echo '    make clean            remove package from dist'

install:
	SRCDIR=${WORKING_DIR} pip3 install -vvvv .

install-yolo34py:
	SRCDIR=${WORKING_DIR} YOLO34PY_FORMAT=True pip3 install -vvvv .

clean:
	pip3 uninstall -y pyyolo

reset: clean install

reset-yolo34py: clean install-yolo34py

build:
	python3 setup.py build_ext --inplace

test:
	python3 -c 'import pyyolo'

.PHONY: all install install-yolo34py clean reset reset-yolo34py build test