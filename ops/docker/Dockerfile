FROM tensorflow/tensorflow:2.2.0-gpu

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y vim wget curl

RUN pip install Pillow==8.3.2
RUN pip install -U scikit-learn

WORKDIR /app/
RUN mkdir image_classification
COPY . image_classification/