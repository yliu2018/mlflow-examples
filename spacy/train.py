import random
from packaging.version import Version

import spacy
from spacy.util import minibatch, compounding

import mlflow.spacy

IS_SPACY_VERSION_NEWER_THAN_OR_EQUAL_TO_3_0_0 = Version(spacy.__version__) >= Version("3.0.0")

from spacy.training.example import Example

# training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]

if __name__ == "__main__":
    # Adaptation of spaCy example to make it work using latest spacy version 3.4.1: https://github.com/mlflow/mlflow/tree/master/examples/spacy

    # create blank model and add ner to the pipeline
    nlp = spacy.blank("en")
    if IS_SPACY_VERSION_NEWER_THAN_OR_EQUAL_TO_3_0_0:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    params = {"n_iter": 100, "drop": 0.5}
    mlflow.log_params(params)

    nlp.begin_training()

    for itn in range(params["n_iter"]):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            example = []
            for i in range(len(texts)):
                doc = nlp.make_doc(texts[i])
                example.append(Example.from_dict(doc, annotations[i]))
            nlp.update(
                example,
                drop=params["drop"],  # dropout - make it harder to memorise data
                losses=losses,
            )
        print("Losses", losses)
        mlflow.log_metrics(losses)

    # Log the spaCy model using mlflow
    mlflow.spacy.log_model(spacy_model=nlp, artifact_path="model")
    model_uri = "runs:/{run_id}/{artifact_path}".format(
        run_id=mlflow.active_run().info.run_id, artifact_path="model"
    )

    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

    # Load the model using mlflow and use it to predict data
    nlp2 = mlflow.spacy.load_model(model_uri=model_uri)
    for text, _ in TRAIN_DATA:
        doc = nlp2(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])