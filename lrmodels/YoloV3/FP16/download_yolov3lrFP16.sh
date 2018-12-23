#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1F6-j-waGDSj5lIWqz88Pff-E6devm5NO" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1F6-j-waGDSj5lIWqz88Pff-E6devm5NO" -o YoloV3FP16.tar.gz
tar -zxvf YoloV3FP16.tar.gz
rm YoloV3FP16.tar.gz

