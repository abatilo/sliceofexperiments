FROM python:3.10 as test-target
WORKDIR /app
COPY test_string_rev.py test_string_rev.py
COPY main.py main.py

FROM python:3.10 as build-target
WORKDIR /app
COPY main.py main.py
ENTRYPOINT ["python", "main.py"]
