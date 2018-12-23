#!/bin/bash
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1qRO7Mf8pS1Urno5z7ZE4U0W9CYTJnk_B" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1qRO7Mf8pS1Urno5z7ZE4U0W9CYTJnk_B" -o tiny-YoloV3FP16.tar.gz
tar -zxvf tiny-YoloV3FP16.tar.gz
rm tiny-YoloV3FP16.tar.gz

