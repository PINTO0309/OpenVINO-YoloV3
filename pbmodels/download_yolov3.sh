curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1rsIhaFS0oA_UsdLsthFi7BLohI90VXmR" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1rsIhaFS0oA_UsdLsthFi7BLohI90VXmR" -o YoloV3.tar.gz
tar -zxvf YoloV3.tar.gz
rm YoloV3.tar.gz

