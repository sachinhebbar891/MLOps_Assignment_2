FROM python:3.8-slim

RUN apt -y update && apt install -y --no-install-recommends \
    libgomp1 \
    && apt clean

COPY . /assignment2

WORKDIR /assignment2

ENV PYTHONPATH "${PYTHONPATH}:/"

RUN apt update && \
  pip install --upgrade pip setuptools && \
  pip install --editable ".[dev]"

EXPOSE 5000

CMD ["flask", "--app", "assignment2", "--debug", "run", "--host", "0.0.0.0"]