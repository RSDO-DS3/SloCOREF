#!/bin/bash

docker run --rm -it --name slo-coref \
  -p 5020:5020 \
  --env COREF_MODEL_PATH="/app/data/bert_based/" \
  --mount type=bind,source="`pwd`/model/cseb_senticoref/",destination="/app/data/bert_based/",ro \
  slo-coref