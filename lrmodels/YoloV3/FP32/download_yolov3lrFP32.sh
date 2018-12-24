#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1U-ilK5qJXZslsiz5wzeywMARKaMU8CpQ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1U-ilK5qJXZslsiz5wzeywMARKaMU8CpQ" -o YoloV3FP32.tar.gz
tar -zxvf YoloV3FP32.tar.gz
rm YoloV3FP32.tar.gz

