import click
import mlflow
import logging
from pathlib import Path
import os
from mlflow.models import ModelSignature
import pandas as pd
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

CONDA_ENV = os.path.join(Path(os.path.dirname(os.path.abspath(__file__))), "conda.yaml")
MODEL_ARTIFACT_PATH = 'inference_pipeline_model'

class InferencePipeline(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import gcld3
        self.detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0,
                                        max_num_bytes=1000)
        self.trained_ner_model = mlflow.spacy.load_model(self.trained_model_uri)
    
    def __init__(self, trained_model_uri, inference_pipeline_uri=None):
        self.trained_model_uri = trained_model_uri
        self.inference_pipeline_uri = inference_pipeline_uri

    def preprocessing_step_lang_detect(self, row):  
        language_detected = self.detector.FindLanguage(text=row[0])
        if language_detected.language != 'en':
            print("found Non-English language text.")
        return language_detected.language

    # for a single row
    def ner_model(self, row):

        # preprocessing: language detection
        language_detected = self.preprocessing_step_lang_detect(row)

        # model inference
        doc = self.trained_ner_model({row[0]})
        pred_entites = [(ent.text, ent.label_) for ent in doc.ents]
        pred_tokens = [(t.text, t.ent_type_, t.ent_iob) for t in doc]

        # postprocessing: add additional metadata
        response = json.dumps({
                'response': {
                    'prediction_entities': pred_entites,
                    'prediction_tokens': pred_tokens
                },
                'metadata': {
                    'language_detected': language_detected,
                },
                'model_metadata': {
                    'trained_model_uri': self.trained_model_uri,
                    'inference_pipeline_model_uri': self.inference_pipeline_uri
                },
            })

        return response
    
    def predict(self, context, model_input):
        results =model_input.apply(self.ner_model, axis=1,  result_type='broadcast')
        return results


# Input and Output formats
input = json.dumps([{'name': 'text', 'type': 'string'}])
output = json.dumps([{'name': 'text', 'type': 'string'}])
# Load model from spec
signature = ModelSignature.from_dict({'inputs': input, 'outputs': output})

@click.command(help="This program creates a multi-step inference pipeline model .")
@click.option("--trained_model_run_id", default=None,
              help="This is the mlflow run id for the trained model")
@click.option("--pipeline_run_name", default="inference_pipeline_model_logging", help="This is the mlflow run name.")
def task(trained_model_run_id, pipeline_run_name):
    with mlflow.start_run(run_name=pipeline_run_name) as mlrun:
        trained_model_uri = f'runs:/{trained_model_run_id}/model'
        inference_pipeline_uri = f'runs:/{mlrun.info.run_id}/{MODEL_ARTIFACT_PATH}'
        mlflow.pyfunc.log_model(artifact_path=MODEL_ARTIFACT_PATH, 
                            conda_env=CONDA_ENV, 
                            python_model=InferencePipeline(trained_model_uri, inference_pipeline_uri), 
                            signature=signature,
                            registered_model_name=MODEL_ARTIFACT_PATH
                            )   
    
        logger.info("finetuned model uri is: %s", trained_model_uri)
        logger.info("inference_pipeline_uri is: %s", inference_pipeline_uri)
        mlflow.log_param("finetuned_model_uri", trained_model_uri)
        mlflow.log_param("inference_pipeline_uri", inference_pipeline_uri)
        mlflow.set_tag('pipeline_step', __file__)

    logger.info("finished logging inference pipeline model")


if __name__ == '__main__':
    task()
