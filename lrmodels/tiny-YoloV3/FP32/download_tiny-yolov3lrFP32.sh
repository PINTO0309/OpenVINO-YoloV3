#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13_5BYWKt8kl9fg-wghep7SYboLV5LAFd" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13_5BYWKt8kl9fg-wghep7SYboLV5LAFd" -o tiny-YoloV3FP32.tar.gz
tar -zxvf tiny-YoloV3FP32.tar.gz
rm tiny-YoloV3FP32.tar.gz

