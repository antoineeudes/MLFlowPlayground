FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# update package manager
RUN apt-get update
RUN apt-get -y upgrade

# install conda
RUN apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN sha256sum Miniconda3-latest-Linux-x86_64.sh
RUN mkdir /root/.conda
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b
RUN rm Miniconda3-latest-Linux-x86_64.sh

# install application
RUN mkdir /app /app/src
WORKDIR /app/
COPY ./requirements.txt /app/
RUN pip install -r requirements.txt
COPY ./src/ /app/src/