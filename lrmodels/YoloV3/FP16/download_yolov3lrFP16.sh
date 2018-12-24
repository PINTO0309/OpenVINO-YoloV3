#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=13DHRI7gho_E3C5tpqQEt7MIz-s6hr5kX" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=13DHRI7gho_E3C5tpqQEt7MIz-s6hr5kX" -o YoloV3FP16.tar.gz
tar -zxvf YoloV3FP16.tar.gz
rm YoloV3FP16.tar.gz

