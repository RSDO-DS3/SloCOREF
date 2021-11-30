FROM python:3.9

WORKDIR /app

COPY ./src /app
COPY ./requirements.txt /app/
COPY ./requirements-api.txt /app/

# For some reason, two versions of torch get installed when installing requirements?
# Installing specific torch version before installing the rest solves this problem.
RUN pip install torch==1.10.0

RUN pip install -r requirements.txt

# Install additional requirements to run REST API
RUN pip install -r requirements-api.txt

EXPOSE 5020

CMD ["uvicorn", "rest_api:app", "--host", "0.0.0.0", "--port", "5020"]