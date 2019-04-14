#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1o8zRe2WteP4S-cRA5sa_mCFsdAeiFIdz" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1o8zRe2WteP4S-cRA5sa_mCFsdAeiFIdz" -o tiny-YoloV3FP16.tar.gz
tar -zxvf tiny-YoloV3FP16.tar.gz
rm tiny-YoloV3FP16.tar.gz

