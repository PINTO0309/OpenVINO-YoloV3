#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1j2PMJbEscSgSGNiflB2tHTLhk1X-U7w_" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1j2PMJbEscSgSGNiflB2tHTLhk1X-U7w_" -o Yolov3-tiny.tar.gz
tar -zxvf Yolov3-tiny.tar.gz
rm Yolov3-tiny.tar.gz

