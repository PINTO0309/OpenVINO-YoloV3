#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=14gUbsl-VJVByUZ4JpzHN64QlYCqaH2QD" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=14gUbsl-VJVByUZ4JpzHN64QlYCqaH2QD" -o tiny-YoloV3.tar.gz
tar -zxvf tiny-YoloV3.tar.gz
rm tiny-YoloV3.tar.gz

