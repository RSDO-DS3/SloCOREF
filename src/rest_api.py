import os
from typing import Optional, List, Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from contextual_model_bert import ContextualControllerBERT
from data import Mention, Token, Document

# To start the REST API, first pip install `fastapi` and `uvicorn`.
# Then, make sure you are in the `src` directory and run something akin to the following (example is for pwsh):
#   $env:COREF_MODEL_PATH="contextual_model_bert\fold0_0"; python -m uvicorn rest_api:app --port=5020

COREF_MODEL_PATH = os.getenv("COREF_MODEL_PATH", None)
if COREF_MODEL_PATH is None:
    raise Exception("Coref model path not specified")

coref_model = ContextualControllerBERT.from_pretrained(COREF_MODEL_PATH)

app = FastAPI(
    title='Coreference resolution API',
    description='Coreference resolution (https://github.com/matejklemen/slovene-coreference-resolution) ' +
                'through a REST API'
)


class _PredictRequestToken(BaseModel):
    text: str
    mention: Optional[int]


class _PredictRequestSentence(BaseModel):
    tokens: List[_PredictRequestToken]


class _PredictRequestBody(BaseModel):
    sentences: List[_PredictRequestSentence]


example_body = _PredictRequestBody(
    sentences=[
        _PredictRequestSentence(tokens=[
            _PredictRequestToken(text='Jože', mention=1),
            _PredictRequestToken(text='je'),
            _PredictRequestToken(text='kupil'),
            _PredictRequestToken(text='sekiro', mention=2),
            _PredictRequestToken(text='.'),
        ]),
        _PredictRequestSentence(tokens=[
            _PredictRequestToken(text='Z'),
            _PredictRequestToken(text='novo', mention=3),
            _PredictRequestToken(text='sekiro', mention=3),
            _PredictRequestToken(text='je'),
            _PredictRequestToken(text='Jože', mention=4),
            _PredictRequestToken(text='posekal'),
            _PredictRequestToken(text='hrast', mention=5),
            _PredictRequestToken(text='.'),
        ])
    ])


@app.post("/predict/coref")
async def predict_coref(
        body: _PredictRequestBody = Body(
            example=example_body,
            default=None,
            media_type='application/json'
        )
):
    doc = __body2doc(body)
    result = coref_model.evaluate_single(doc)
    return result


def __body2doc(body: _PredictRequestBody):
    tokens = {}
    sentences = []
    mentions = {}
    clusters = []

    document_idx = 0
    sentence_idx = 0
    for _sentence in body.sentences:
        token_idx = 0
        sentences.append([])
        for _token in _sentence.tokens:
            token_id = str(sentence_idx + 1) + "-" + str(token_idx + 1)
            # Note: MSD and lemma fields are not used in BERT model, so we are passing a None.
            tokens[token_id] = Token(token_id, _token.text, None, None, sentence_idx, token_idx,
                                     document_idx)

            if _token.mention is not None:
                mention_id = "M-" + str(_token.mention)

                if mention_id not in mentions:
                    mentions[mention_id] = Mention(mention_id, [])
                    clusters.append([mention_id])
                mentions[mention_id].tokens.append(tokens[token_id])

            sentences[sentence_idx].append(token_id)

            token_idx += 1
            document_idx += 1
        sentence_idx += 1

    doc = Document("D-1", tokens, sentences, mentions, clusters)
    __sanity_check(doc)
    return doc


def __sanity_check(doc: Document):
    # Make sure that no. of clusters and mentions matches
    if len(doc.clusters) != len(doc.mentions):
        raise HTTPException(status_code=500, detail="Number of mentions and clusters does not match.")

    # Make sure that tokens in each mention are actually together and in the same sentence.
    for mention in doc.mentions.values():
        sentence_idx = None
        token_idx = None
        for token in mention.tokens:
            if sentence_idx is None:
                sentence_idx = token.sentence_index
                token_idx = token.position_in_sentence
            else:
                if sentence_idx != token.sentence_index:
                    raise HTTPException(status_code=500, detail="Mention tokens are not in the same sentence.")

                if token_idx != token.position_in_sentence - 1:
                    raise HTTPException(status_code=500, detail="Mention tokens are not consecutive.")

                token_idx = token.position_in_sentence
