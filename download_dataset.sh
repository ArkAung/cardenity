#!/bin/bash

# Download images
wget http://imagenet.stanford.edu/internal/car196/car_ims.tgz -P datasets

# Download annotations
wget http://imagenet.stanford.edu/internal/car196/cars_annos.mat -P datasets

# Untar images tgz archive
cd datasets && tar -xvzf car_ims.tgz
rm datasets/car_ims.tgz