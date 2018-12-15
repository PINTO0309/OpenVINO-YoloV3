curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1PMowka0ncBM0aqFHyXQthmUBn2Slou3u" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1PMowka0ncBM0aqFHyXQthmUBn2Slou3u" -o YoloV3FP16.tar.gz
tar -zxvf YoloV3FP16.tar.gz
rm YoloV3FP16.tar.gz

