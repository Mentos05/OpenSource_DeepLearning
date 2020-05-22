# GPU-accelerated Docker container for Deep Learning
<p>An efficient way to develop deep learning applications.</p>
<p>Highly focussed on running Computer Vision applications like this one:.<p>
<img src="misc/cv.png" alt="Computer Vision Example" align='left' style="width: 40%; height: 100%"/> <br clear='left'>

## Overview
This repository provides ressources to spin up a GPU-accelerated docker
container including:
* Jupyter Lab
* Python 3.7 environment
* various Open Source libraries (details in [.yml-File](https://github.com/Mentos05/OpenSource_DeepLearning/blob/master/GPU%20accelerated%20docker%20container/jupyterlab_environments/python_37.yml))
* OpenCV, OpenPose and YOLOv4

You have two options:
1. Pull the container from my docker-repository from hub.docker.com (easy but not customizable)
2. Build your own container (harder but customizable)

## Requirements
* System with NVIDIA GPU (tested with RTX 2070)
* Linux OS (tested with Ubuntu 18.04)
* NVIDIA Driver (recommend latest one)
* NVIDIA CUDA 10.2 or higher
* NVIDIA Docker (https://github.com/NVIDIA/nvidia-docker)

## Container Setup
### Option 1 - Pull the container from my repository
0. Login to hub.docker.com: `docker login --username your-user-name --password your-password`
1. Pull the image: `docker pull michaelgorkow/main_repository:opensource_deeplearning`
2. Run the image: `docker run -it --net=host --gpus all michaelgorkow/main_repository:opensource_deeplearning`

### Option 2 - Build your own container
Copy the following files/folder into a folder:
* Dockerfile
* jupyterlab_environments

Go into this folder and run: 
`docker build .  -t deeplearning:gpu`

This will create a docker container using nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as base image.<br>

You can then run this container by simply executing: `docker run -it --net=host --gpus all deeplearning:gpu`

## Run Options
The container uses environment variables to setup ESP & JupyterLab during the run.<br>
You can make changes to these variables by adding `-e ENVIRONMENT_VARIABLE VALUE` to it.<br>
The following variables are available:<br>

| Variable | Description | Default |
| ------ | ------ | ------ |
| JUPYTERLAB_PORT | JupyterLab port | 8080 |
| JUPYTERLAB_NBDIR | JupyterLab notebook directory | /data/notebooks/ |

Example: `docker run -it --net=host -e JUPYTERLAB_PORT 8000 --gpus all deeplearning:gpu` will run Jupyter Lab on port 8000.

## Jupyter Lab
Open the following URL in your browser:
* JupyterLab: localhost:8080

## Jupyter Lab (Python Environment)
Some of the Python packages installed are:<br>

| Package | Description |
| ------ | ------ |
| TensorFlow | [TensorFlow](https://github.com/tensorflow/tensorflow) |
| OpenCV | [OpenCV](https://github.com/opencv/opencv) |
| OpenPose | [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) |
| Darknet | [Darknet](https://github.com/AlexeyAB/darknet) |

For a full list, please look at the [python_37.yml](https://github.com/Mentos05/OpenSource_DeepLearning/blob/master/GPU%20accelerated%20docker%20container/jupyterlab_environments/python_37.yml)-file.
When building your own container, you can add your own environment.yml files into /jupyterlab_environments folder to have customized Python environments.

## Tips & Tricks
### Share ressources with your container
If you want to share ressources with your container, e.g. a webcam, you can do so by adapting your docker run command.<br>
To share devices, use `docker run --device=/dev/video0:/dev/video0 --net=host deeplearning:gpu` which will share your webcam (video0)<br>
To share a folder, e.g. with additional data like models, projects, etc. use `docker run -v folder-on-host:folder-on-container --net=host deeplearning:gpu`

Example: For my needs I usually start my container with the following command to share my local notebooks, my webcam, hosts networking interface and to allow GUI applications (e.g. Opencv).<br>
```
docker run -it --privileged=true --net=host --ipc=host \
           -v /home/michael/Development/github.com/:/data/notebooks \
           --device=/dev/video0:/dev/video0 \
           --gpus all -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix -v /var/run/dbus:/var/run/dbus \
           michaelgorkow/main_repository:opensource_deeplearning
```

### Run GUI applications inside your container
I am using OpenCV very often to display the scored images from ESP. To allow OpenCV to access your hosts display you'll have to allow access to your X server.
To do this simply type `xhost +` on your host system.
Additionally you'll have to provide some information to your container by adding the following statements to your run-command:<br>
```
-e DISPLAY=$DISPLAY
-v /tmp/.X11-unix:/tmp/.X11-unix
```

### Verify/Monitor GPU Usage
While you should notice a significant performance improvement while training/scoring your deep learning models you can also monitor GPU usage by using `watch -n 1 nvidia-smi`.
Make sure you run this command on your host, not inside the container.

## Feedback
I've put a lot of effort in this container setup to support the creation of beautiful deep learning examples.
However noone is perfect. So if you encounter any problems please let me know.
Also if you think I am missing important things like a supercool python package, feel free to reach out to me and I'll include it. But be aware that I do this in my freetime. ;)
