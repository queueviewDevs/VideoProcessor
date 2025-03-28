FROM python:3.13.2-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --upgrade pip==25.0.1
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

CMD ["python", "face_detection.py"]