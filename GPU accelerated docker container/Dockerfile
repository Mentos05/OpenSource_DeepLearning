FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
MAINTAINER Michael Gorkow <michael.gorkow@sas.com>

#Install ESP dependencies
#RUN yum -y update && yum -y install wget unzip numactl java-1.8.0-openjdk NetworkManager epel-release
RUN apt-get update -y &&  apt-get install -y wget git unzip libssl-dev default-jdk python-pip python3 python3-apt software-properties-common && \
    pip install numpy

#Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

#Create Anaconda Environment with Python 3.7 + Jupyter Lab + various packages
ENV PATH $PATH:/opt/conda/bin
RUN conda update --all -y && conda install nb_conda_kernels cython conda-build -y
ADD jupyterlab_environments /opt/jupyterlab_environments
RUN for env_file in /opt/jupyterlab_environments/*; do conda env create -f $env_file; done

#Install CMake version 3.17 (needed for OpenPose compiling)
RUN mkdir /opt/cmake && wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz -P /opt/cmake && \
    tar -xzf /opt/cmake/cmake-3.17.2.tar.gz -C /opt/cmake/
WORKDIR /opt/cmake/cmake-3.17.2
RUN ./bootstrap && make -j$(nproc) && make install -j$(nproc)

#Install OpenCV from Source for extended codec support (video saving) -> uses CUDA support
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt-get install -y build-essential checkinstall pkg-config yasm gfortran libjpeg8-dev libpng-dev \
    software-properties-common libgtk2.0-dev libtbb-dev qt5-default libjasper1 libtiff-dev libavcodec-dev \ 
    libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev \
    libopencore-amrwb-dev libavresample-dev x264 v4l-utils libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev libgphoto2-dev libeigen3-dev libhdf5-dev doxygen libgtk2.0-dev
Run git clone https://github.com/opencv/opencv.git /opt/opencv/ && \
    git clone https://github.com/opencv/opencv_contrib.git /opt/opencv_contrib/ && \
    mkdir /opt/opencv/build
WORKDIR /opt/opencv/build
Run /usr/local/bin/cmake -DCMAKE_CXX_FLAGS="-fPIC " \
           		   -DPYTHON3_PACKAGES_PATH="/opt/conda/envs/python_37/lib/python3.7/site-packages" \
			   -DPYTHON3_LIBRARY="/opt/conda/envs/python_37/lib/libpython3.7m.a" \
			   -DPYTHON3_NUMPY_INCLUDE_DIRS="/opt/conda/envs/python_37/lib/python3.7/site-packages/numpy/core/include" \
			   -DPYTHON3_EXECUTABLE="/opt/conda/envs/python_37/bin/python" \
			   -DPYTHON3_INCLUDE_DIR="/opt/conda/envs/python_37/include/python3.7m" \
			   -DOPENCV_GENERATE_PKGCONFIG="1" \
			   -DOPENCV_EXTRA_MODULES_PATH="/opt/opencv_contrib/modules" \
			   -DWITH_FFMPEG=1 \
			   -DWITH_GSTREAMER="1" \
			   -DWITH_CUDA="1" \
			   -DWITH_CUBLAS="1" \
			   -DCUDA_FAST_MATH="1" \
			   -DENABLE_FAST_MATH="1" ..
Run make all -j$(nproc) && make install -j$(nproc)

#Install CMU OpenPose from Source https://github.com/CMU-Perceptual-Computing-Lab/openpose
RUN apt-get install -y liblmdb-dev libopenblas-dev libboost-dev libboost-all-dev
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose /opt/openpose/ && mkdir /opt/openpose/build
# Name changes in OpenCV v4 ...
#RUN sed -i "s|CV_LOAD_IMAGE_COLOR|cv::IMREAD_COLOR|g" /opt/openpose/3rdparty/caffe/src/caffe/layers/window_data_layer.cpp && \
#    sed -i "s|CV_LOAD_IMAGE_COLOR|cv::IMREAD_COLOR|g" /opt/openpose/3rdparty/caffe/src/caffe/util/io.cpp && \
#    sed -i "s|CV_LOAD_IMAGE_GRAYSCALE|cv::IMREAD_GRAYSCALE|g" /opt/openpose/3rdparty/caffe/src/caffe/util/io.cpp && \
#    sed -i "s|CV_LOAD_IMAGE_GRAYSCALE|cv::IMREAD_GRAYSCALE|g" /opt/openpose/examples/tests/resizeTest.cpp
WORKDIR /opt/openpose/build
RUN /usr/local/bin/cmake -DBUILD_CAFFE=1 \
                           -DPYTHON3_PACKAGES_PATH="/opt/conda/envs/python_37/lib/python3.7/site-packages" \
			   -DPYTHON3_LIBRARY="/opt/conda/envs/python_37/lib/libpython3.7m.a" \
			   -DPYTHON3_NUMPY_INCLUDE_DIRS="/opt/conda/envs/python_37/lib/python3.7/site-packages/numpy/core/include" \
			   -DPYTHON3_EXECUTABLE="/opt/conda/envs/python_37/bin/python" \
			   -DPYTHON3_INCLUDE_DIR="/opt/conda/envs/python_37/include/python3.7m" \
			   -DCMAKE_CXX_FLAGS="-fPIC" \
			   -DBUILD_PYTHON=1 \
			   -DCMAKE_INSTALL_PREFIX="/opt/openpose/build/install/" ..
RUN make all -j$(nproc) && \
    make install -j$(nproc)
RUN conda-develop -n python_37 /opt/openpose/build/install/python

#Install YOLOv4 from Source https://github.com/AlexeyAB/darknet.git
RUN git clone https://github.com/AlexeyAB/darknet.git /opt/darknet/
WORKDIR /opt/darknet
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH && /opt/darknet/build.sh && \
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -O /opt/darknet/yolov4.weights && \
    sed -i "s|./libdarknet.so|/opt/darknet/libdark.so|g" /opt/darknet/darknet.py && \
    sed -i "s|data/coco.names|/opt/darknet/data/coco.names|g" /opt/darknet/cfg/coco.data && \
    conda-develop -n python_37 /opt/darknet/

RUN mkdir -p /data/notebooks/

#Set environment variables
ENV JUPYTERLAB_PORT=8080
ENV JUPYTERLAB_NBDIR=/data/notebooks/

#Create start script
RUN echo -e '#!/bin/bash\n' \
            'jupyter lab --port $JUPYTERLAB_PORT --ip 0.0.0.0 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.notebook_dir=$JUPYTERLAB_NBDIR --NotebookApp.allow_origin='*'\n' \
            'while true \n' \
            'do \n' \
            'sleep 3600 \n' \
            'done'> /opt/start.sh
RUN chmod +x /opt/start.sh

# Calls start script
CMD /opt/start.sh
