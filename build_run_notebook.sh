#!/bin/bash

nvidia-docker build -f Dockerfile.jupyter . -t sommet/notebook
nvidia-docker run -p8890:8890 -it -v/scratch/:/scratch -v/home/nina/:/home/nina sommet/notebook
