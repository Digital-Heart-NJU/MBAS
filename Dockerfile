## Pull from existing image
FROM pytorch/pytorch:latest
FROM python:3.9.13

## Copy requirements
COPY ./requirements.txt .

## Install Python packages in Docker image
RUN pip install -r requirements.txt

## Copy all files
COPY ./ ./

## Execute the inference command 
ENTRYPOINT python -m predict

