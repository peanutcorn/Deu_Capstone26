FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

MAINTAINER "parkms@markany.com"

RUN apt-get update -y && \
	apt-get -y install libgl1-mesa-glx && \
	apt-get -y install libglib2.0-0

RUN	apt-get -y install vim
		
COPY . /pose

WORKDIR /pose

RUN pip install -r requirements.txt

RUN cd lib && \
	make && \
	cd ..
