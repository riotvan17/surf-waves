FROM python:3.8.13-slim-buster
COPY api /api
COPY requirements.txt /requirements.txt
COPY utils.py /utils.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
