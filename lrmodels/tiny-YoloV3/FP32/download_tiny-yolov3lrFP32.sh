#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1OyLOcayAh1Ia0XhbusM9H5WvT7faoBHY" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1OyLOcayAh1Ia0XhbusM9H5WvT7faoBHY" -o tiny-YoloV3FP32.tar.gz
tar -zxvf tiny-YoloV3FP32.tar.gz
rm tiny-YoloV3FP32.tar.gz

