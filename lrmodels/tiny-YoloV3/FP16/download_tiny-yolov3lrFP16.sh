#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1qYTE1fEwSduDYrv6f3J_3GKbkio5m3E2" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1qYTE1fEwSduDYrv6f3J_3GKbkio5m3E2" -o tiny-YoloV3FP16.tar.gz
tar -zxvf tiny-YoloV3FP16.tar.gz
rm tiny-YoloV3FP16.tar.gz

