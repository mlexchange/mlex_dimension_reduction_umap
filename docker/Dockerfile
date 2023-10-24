FROM python:3.9
LABEL maintainer="THE MLEXCHANGE TEAM"

COPY docker/requirements.txt requirements.txt 

RUN apt-get update && apt-get install -y \
    tree 

RUN pip3 install --upgrade pip &&\
    pip3 install -r requirements.txt 

WORKDIR /app/work
ENV HOME /app/work
ENV PYTHONUNBUFFERED=1

COPY umap_run.py umap_run.py
COPY utils.py utils.py
CMD ["echo", "running umap"]