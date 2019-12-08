FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /usr/src/GTSRB

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "./main.py" ]