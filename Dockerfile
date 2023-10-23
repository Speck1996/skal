FROM tensorflow/tensorflow:2.13.0-gpu

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install libcublas-12-0 -y && \
    apt-get install -y git

RUN useradd -ms /bin/bash speck

USER speck

WORKDIR /skal

ENTRYPOINT [ "/bin/bash" ]