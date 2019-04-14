#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1NW3wkz92aMYcxynC1is44IdUQsJ9OW9G" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1NW3wkz92aMYcxynC1is44IdUQsJ9OW9G" -o YoloV3FP16.tar.gz
tar -zxvf YoloV3FP16.tar.gz
rm YoloV3FP16.tar.gz

