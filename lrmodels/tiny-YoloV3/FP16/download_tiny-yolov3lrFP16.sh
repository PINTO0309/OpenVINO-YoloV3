#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1hIDFhX7-80xKCoxHNpNLv4Vnjrh0i1s1" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1hIDFhX7-80xKCoxHNpNLv4Vnjrh0i1s1" -o tiny-YoloV3FP16.tar.gz
tar -zxvf tiny-YoloV3FP16.tar.gz
rm tiny-YoloV3FP16.tar.gz

