# slo-coref REST API

Rest API is run via FastAPI/uvicorn.

## Installation

Install base and API requirements by running 
```sh
$ pip install -r requirements.txt
$ pip install -r requirements-api.txt
```

## Setup & running

Running REST API requires two environment variables:
- `COREF_MODEL_PATH`, containing the path to the coref model
- `CLASSLA_RESOURCES_DIR`, a path to where the CLASSLA resources will be downloaded (if are not already) and used


REST API can then be ran by moving into the `src` directory and running the `uvicorn` module:

```sh
$ cd ./src
$ python -m uvicorn rest_api:app --port=5020
```

You can, of course, define required env variables on the fly when running, for example:

```sh
$ CLASSLA_RESOURCES_DIR=.\classla_resources \
  COREF_MODEL_PATH=.\contextual_model_bert \
  python -m uvicorn rest_api:app --port=5020
```

Assuming everything went smoothly, the API will become available at http://localhost:5020/predict/coref.
```
...
INFO:     Uvicorn running on http://127.0.0.1:5020 (Press CTRL+C to quit)
```

To test it, try sending a request with **curl**:
```sh
$ curl -X POST -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"threshold": 0.60, "return_singletons": true, "text": "Janez Novak je šel v Mercator. Tam je kupil mleko. Nato ga je spreletela misel, da bi moral iti v Hofer."}' \
  "http://localhost:5020/predict/coref"
```

### Documentation

After starting up the API, the OpenAPI/Swagger documentation will also become accessible at http://localhost:5020/docs.

## Docker

### Building

To build the docker image, run 

```sh
docker build --tag slo-coref -f Dockerfile .
```

### Running

To run the docker image, run the following command with properly fixed mount `source` paths:

```sh
docker run --rm -it --name slo-coref \
  -p 5020:5020 \
  --env COREF_MODEL_PATH="/app/data/bert_based/" \
  --mount type=bind,source="/path/to/contextual_model_bert/",destination="/app/data/bert_based/",ro \
  slo-coref
```