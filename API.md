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
