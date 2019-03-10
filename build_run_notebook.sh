#!/bin/bash

nvidia-docker build -f docker/Dockerfile.jupyter . -t sommet/notebook
nvidia-docker run -p8888:8888 -it -v/scratch/:/scratch -v/home/nina:/home/nina sommet/notebook
