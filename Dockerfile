FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip --default-timeout=1000 install -r requirements.txt
CMD ["python3","inference.py"]