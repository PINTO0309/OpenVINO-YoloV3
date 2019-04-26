#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1LABAA3HdFWUdmav45Szy3kRkV-7ZYgHl" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1LABAA3HdFWUdmav45Szy3kRkV-7ZYgHl" -o YoloV3FP32.tar.gz
tar -zxvf YoloV3FP32.tar.gz
rm YoloV3FP32.tar.gz

