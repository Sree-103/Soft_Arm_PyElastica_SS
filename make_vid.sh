#!/bin/bash


ffmpeg -threads 8 -r 15 -i images/moving_arm%03d.png -b:v 90M -vcodec mpeg4 ./output_video.mp4
