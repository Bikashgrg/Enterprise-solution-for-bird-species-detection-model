FROM ubuntu
FROM python:3.7.10
FROM tensorflow/tensorflow:2.8.0

ADD FlaskObjectDetection /FlaskObjectDetection

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install Flask
RUN pip install tensorflow-serving-api
RUN pip install grpcio
RUN pip install Flask-WTF
RUN pip install numpy
RUN pip install opencv-python
RUN pip install Pillow
RUN pip install matplotlib

WORKDIR "./FlaskObjectDetection"
CMD ["python", "./app.py"]
