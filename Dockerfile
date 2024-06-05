FROM python:3.9
LABEL maintainer="THE MLEXCHANGE TEAM"

RUN apt-get update
RUN pip3 install --upgrade pip &&\
    pip3 install .

WORKDIR /app/work
ENV HOME /app/work
ENV PYTHONUNBUFFERED=1

COPY umap_run.py umap_run.py
COPY src src
CMD ["echo", "running umap"]
