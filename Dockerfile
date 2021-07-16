FROM python:3.9

WORKDIR /app

COPY ./src /app
COPY ./requirements.txt /app/

# For some reason, two versions of torch get installed when installing requirements (1.9.0 and 1.8.1).
# Installing specific torch version before installing the rest solves this problem.
RUN pip install torch==1.8.1

RUN pip install -r requirements.txt
RUN pip install fastapi uvicorn

EXPOSE 5020

CMD ["uvicorn", "rest_api:app", "--host", "0.0.0.0", "--port", "5020"]