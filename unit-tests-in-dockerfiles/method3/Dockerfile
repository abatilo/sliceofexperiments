FROM python:3.10

WORKDIR /app
COPY test_string_rev.py test_string_rev.py
COPY main.py main.py
RUN python -m unittest test_string_rev

ENTRYPOINT ["python", "main.py"]
