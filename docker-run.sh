#!/bin/bash

docker run \
  -d \
  -it \
  --name slo-coref \
  -p 5002:5020 \
  --restart always \
  --env COREF_MODEL_PATH="/app/data/bert_based/" \
  --mount type=bind,source="`pwd`/model/cseb_senticoref/",destination="/app/data/bert_based/",ro \
  slo-coref