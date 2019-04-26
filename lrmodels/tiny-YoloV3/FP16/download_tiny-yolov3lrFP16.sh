#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1IThHGY0Dt5ZFAB_IBwoKaWgf-phoDgmv" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1IThHGY0Dt5ZFAB_IBwoKaWgf-phoDgmv" -o tiny-YoloV3FP16.tar.gz
tar -zxvf tiny-YoloV3FP16.tar.gz
rm tiny-YoloV3FP16.tar.gz

