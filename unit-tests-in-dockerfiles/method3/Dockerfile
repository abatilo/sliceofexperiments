FROM python:3.10

WORKDIR /app
COPY test_string_rev.py test_string_rev.py
RUN python -m unittest test_string_rev

COPY main.py main.py

ENTRYPOINT ["python", "main.py"]
