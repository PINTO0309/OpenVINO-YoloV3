curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Kbw_YEsoc3hCRN7jNsB6JI3bMKrPeXkB" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Kbw_YEsoc3hCRN7jNsB6JI3bMKrPeXkB" -o YoloV3FP16.tar.gz
tar -zxvf YoloV3FP16.tar.gz
rm YoloV3FP16.tar.gz

