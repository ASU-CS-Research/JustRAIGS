FROM ubuntu:22.04
#WORKDIR /root/

# Add libcuda dummy dependency
#ADD control .
#RUN apt-get update && \
#	DEBIAN_FRONTEND=noninteractive apt-get install --yes equivs && \
#	equivs-build control && \
#	dpkg -i libcuda1-dummy_11.8_all.deb && \
#	rm control libcuda1-dummy_11.8* && \
#	apt-get remove --yes --purge --autoremove equivs && \
#	rm -rf /var/lib/apt/lists/*

## Setup Lambda repository
#ADD lambda.gpg .
#RUN apt-get update && \
#	apt-get install --yes gnupg && \
#	gpg --dearmor -o /etc/apt/trusted.gpg.d/lambda.gpg < lambda.gpg && \
#	rm lambda.gpg && \
#	echo "deb http://archive.lambdalabs.com/ubuntu jammy main" > /etc/apt/sources.list.d/lambda.list && \
#	echo "Package: *" > /etc/apt/preferences.d/lambda && \
#	echo "Pin: origin archive.lambdalabs.com" >> /etc/apt/preferences.d/lambda && \
#	echo "Pin-Priority: 1001" >> /etc/apt/preferences.d/lambda && \
#	echo "cudnn cudnn/license_preseed select ACCEPT" | debconf-set-selections && \
#	apt-get update && \
#	DEBIAN_FRONTEND=noninteractive \
#		apt-get install \
#		--yes \
#		--no-install-recommends \
#		--option "Acquire::http::No-Cache=true" \
#		--option "Acquire::http::Pipeline-Depth=0" \
#		lambda-stack-cuda \
#		lambda-server && \
#	rm -rf /var/lib/apt/lists/*
#
## Setup for nvidia-docker
#ENV NVIDIA_VISIBLE_DEVICES all
#ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
#ENV NVIDIA_REQUIRE_CUDA "cuda>=11.8"

RUN apt update && apt install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

WORKDIR /opt/app
COPY requirements.txt /opt/app

RUN pip install --upgrade pip setuptools wheel
RUN pip install h5py --only-binary=h5py
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install SimpleITK

ENV PYTHONPATH "${PYTHONPATH}:/opt/app"

#COPY . /opt/app
COPY ./src/inference/saved_models/ /opt/app/src/inference/saved_models/
COPY ./src/inference/__init__.py /opt/app/src/inference/__init__.py
COPY ./src/inference/control /opt/app/src/inference/control
COPY ./src/inference/helper.py /opt/app/src/inference/helper.py
COPY ./src/inference/inference.py /opt/app/src/inference/inference.py
COPY ./src/inference/lambda.gpg /opt/app/src/inference/lambda.gpg
COPY ./test/ /opt/app/test/

#COPY ./src/inference/helper.py /opt/app/src/inference/helper.py
#COPY ./src/inference/inference.py /opt/app/src/inference/inference.py
#COPY ./test/ /opt/app/test/

ENTRYPOINT ["python3", "src/inference/inference.py"]
#ENTRYPOINT ["python3"]
