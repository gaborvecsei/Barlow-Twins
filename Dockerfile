FROM tensorflow/tensorflow:2.4.2-gpu

WORKDIR /

COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt

CMD ["/bin/bash"]

