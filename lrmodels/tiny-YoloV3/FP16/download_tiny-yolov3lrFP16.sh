curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Fd55eIwUECRlOrkDkH3cC92Paz0o5Mkn" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Fd55eIwUECRlOrkDkH3cC92Paz0o5Mkn" -o tiny-YoloV3FP16.tar.gz
tar -zxvf tiny-YoloV3FP16.tar.gz
rm tiny-YoloV3FP16.tar.gz

