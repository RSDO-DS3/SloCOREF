# slo-coref REST API

Rest API is run via FastAPI/uvicorn.

## Installation

If you have not yet, install `requirements-api.txt` by running 
```sh
pip install -r requirements-api.txt
```

## Setup & running

Running REST API requires two environment variables:
- `COREF_MODEL_PATH`, containing the path to the coref model
- `CLASSLA_RESOURCES_DIR`, a path to where the CLASSLA resources should be downloaded


REST API can then be ran by moving into the `src` directory and running the `uvicorn` module:

```sh
cd ./src
python -m uvicorn rest_api:app --port=5020
```

You can, of course, define required env variables on the fly, for example:

```sh
CLASSLA_RESOURCES_DIR=.\classla_resources COREF_MODEL_PATH=.\contextual_model_bert\fold0_0 python -m uvicorn rest_api:app --port=5020
```

## Docker

### Building

To build the docker image, run 

```sh
docker build --tag slo-coref -f Dockerfile .
```

### Running

To run the docker image, run the following command with properly fixed mount `source` paths:

```sh
docker run --rm -it --name slo-coref -p 5020:5020 --env CLASSLA_RESOURCES_DIR="/app/data/classla" --env COREF_MODEL_PATH="/app/data/bert_based/" --mount type=bind,source="/path/to/contextual_model_bert/",destination="/app/data/bert_based/",ro --mount type=bind,source="/path/to/classla_resources",destination="/app/data/classla/" slo-coref
```