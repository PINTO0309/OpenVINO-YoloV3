curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1uTXsPJGM5zMGU5OnCJl6d2yft7yzRP3c" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1uTXsPJGM5zMGU5OnCJl6d2yft7yzRP3c" -o YoloV3FP32.tar.gz
tar -zxvf YoloV3FP32.tar.gz
rm YoloV3FP32.tar.gz

