ARG TENSORFLOW_VERSION=latest
FROM tensorflow/tensorflow:${TENSORFLOW_VERSION}

USER root

EXPOSE 8888
CMD ["jupyter notebook"]

COPY requirements.txt .
COPY emnist-network.ipynb .
#COPY start_notebook.sh .

RUN pip install -r requirements.txt
