import json

from src.contextual_model_bert import ContextualControllerBERT
from src.data import Mention, Token, Document

# A model I've trained, I think it was on coref149. No idea how smart it is.
bert_model = ContextualControllerBERT.from_pretrained("contextual_model_bert\\fold0_0")

# json output of sample text, produced by https://github.com/clarinsi/classla
sample_text_json = json.loads("""[
  [
    [
      {
        "id": 1,
        "text": "Jože",
        "lemma": "Jože",
        "upos": "PROPN",
        "xpos": "Npmsn",
        "feats": "Case=Nom|Gender=Masc|Number=Sing"
      },
      {
        "id": 2,
        "text": "je",
        "lemma": "biti",
        "upos": "AUX",
        "xpos": "Va-r3s-n",
        "feats": "Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin"
      },
      {
        "id": 3,
        "text": "kupil",
        "lemma": "kupiti",
        "upos": "VERB",
        "xpos": "Vmep-sm",
        "feats": "Aspect=Perf|Gender=Masc|Number=Sing|VerbForm=Part"
      },
      {
        "id": 4,
        "text": "novo",
        "lemma": "nov",
        "upos": "ADJ",
        "xpos": "Agpfsa",
        "feats": "Case=Acc|Degree=Pos|Gender=Fem|Number=Sing"
      },
      {
        "id": 5,
        "text": "sekiro",
        "lemma": "sekira",
        "upos": "NOUN",
        "xpos": "Ncfsa",
        "feats": "Case=Acc|Gender=Fem|Number=Sing",
        "misc": "SpaceAfter=No"
      },
      {
        "id": 6,
        "text": ".",
        "lemma": ".",
        "upos": "PUNCT",
        "xpos": "Z"
      }
    ]
  ],
  [
    [
      {
        "id": 1,
        "text": "S",
        "lemma": "z",
        "upos": "ADP",
        "xpos": "Si",
        "feats": "Case=Ins"
      },
      {
        "id": 2,
        "text": "to",
        "lemma": "ta",
        "upos": "DET",
        "xpos": "Pd-fsi",
        "feats": "Case=Ins|Gender=Fem|Number=Sing|PronType=Dem"
      },
      {
        "id": 3,
        "text": "sekiro",
        "lemma": "sekira",
        "upos": "NOUN",
        "xpos": "Ncfsi",
        "feats": "Case=Ins|Gender=Fem|Number=Sing"
      },
      {
        "id": 4,
        "text": "je",
        "lemma": "biti",
        "upos": "AUX",
        "xpos": "Va-r3s-n",
        "feats": "Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres|VerbForm=Fin"
      },
      {
        "id": 5,
        "text": "Jože",
        "lemma": "Jože",
        "upos": "PROPN",
        "xpos": "Npmsn",
        "feats": "Case=Nom|Gender=Masc|Number=Sing"
      },
      {
        "id": 6,
        "text": "posekal",
        "lemma": "posekati",
        "upos": "VERB",
        "xpos": "Vmep-sm",
        "feats": "Aspect=Perf|Gender=Masc|Number=Sing|VerbForm=Part"
      },
      {
        "id": 7,
        "text": "drevo",
        "lemma": "drevo",
        "upos": "NOUN",
        "xpos": "Ncnsa",
        "feats": "Case=Acc|Gender=Neut|Number=Sing",
        "misc": "SpaceAfter=No"
      },
      {
        "id": 8,
        "text": ".",
        "lemma": ".",
        "upos": "PUNCT",
        "xpos": "Z"
      }
    ]
  ]
]""")

sample_tokens = {}
sample_sentences = []

document_idx = 0
sentence_idx = 0
for sentence in sample_text_json:
    token_idx = 0
    sample_sentences.append([])
    for token in sentence[0]:
        token_id = str(sentence_idx + 1) + "-" + str(token_idx + 1)
        # TODO Not sure token["xpos"] is the correct contents for the MSD field.
        sample_tokens[token_id] = Token(token_id, token["text"], token["lemma"], token["xpos"], sentence_idx, token_idx, document_idx)
        sample_sentences[sentence_idx].append(token_id)

        token_idx += 1
        document_idx += 1
    sentence_idx += 1

# Automate extraction of mentions from tokens in some way.
sample_mentions = {
    "M1": Mention("M1", [sample_tokens["1-1"]]),
    "M2": Mention("M2", [sample_tokens["1-4"], sample_tokens["1-5"]]),
    "M3": Mention("M3", [sample_tokens["2-3"]]),
    "M4": Mention("M4", [sample_tokens["2-5"]]),
    "M5": Mention("M5", [sample_tokens["2-7"]]),
}

# TODO But I actually "don't know" what the gt clusters are?
sample_clusters = [["M1", "M4"], ["M2", "M3"], ["M5"]]

# Prepare a document object for evaluation
sample_doc = Document("D1", sample_tokens, sample_sentences, sample_mentions, sample_clusters)

result = bert_model.evaluate_single(sample_doc)

print(result)
