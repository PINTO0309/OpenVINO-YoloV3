curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=1Rpy7jPrxRm4NuJdJQHL3nqFjM0TuQQMJ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1Rpy7jPrxRm4NuJdJQHL3nqFjM0TuQQMJ" -o tiny-YoloV3.tar.gz
tar -zxvf tiny-YoloV3.tar.gz
rm tiny-YoloV3.tar.gz

