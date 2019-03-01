#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=17rhAPwZOW9x_dL3vL5Fo1SsIJlnKYxHh" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=17rhAPwZOW9x_dL3vL5Fo1SsIJlnKYxHh" -o Yolo3.tar.gz
tar -zxvf Yolo3.tar.gz
rm Yolo3.tar.gz

