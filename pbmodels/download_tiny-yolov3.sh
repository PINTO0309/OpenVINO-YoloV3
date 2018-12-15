curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Slnh3ShPOog1En5rkWi5NUQaVhRp7M2M" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Slnh3ShPOog1En5rkWi5NUQaVhRp7M2M" -o tiny-YoloV3.tar.gz
tar -zxvf tiny-YoloV3.tar.gz
rm tiny-YoloV3.tar.gz

