import os
from dotenv import load_dotenv

from ml_models.bitcoin.bitcoin_multi_label_model import BitcoinMultiLabelModel
from ml_models.sentiment.sentiment_model import SentimentModel
from celery import Celery
from celery.utils.log import get_task_logger

load_dotenv()
USER = os.environ.get("RABBIT_USER")
PASSWORD = os.environ.get("RABBIT_PASSWORD")
BROKER = 'localhost'
# BROKER = os.environ.get("RABBIT_BROKER")

BROKER_URL = f'amqp://{USER}:{PASSWORD}@{BROKER}:5672'
# LOCAL_MODEL_PATH = '/home/models'

celery = Celery("classification_queue", broker=BROKER_URL)
logger = get_task_logger(__name__)

model_name = os.environ.get('model_name')


if model_name == 'sentiment_model':
    model = SentimentModel(path_to_weights='/src/nlp/model_weights/sentiment_model')
elif model_name == 'bitcoin_multi_label':
    model = BitcoinMultiLabelModel(path_to_weights='nlp/model_weights/bitcoin/multi-label-both.pt')


@celery.task
def classify(text):

    sentiment = model.predict(text)

    return sentiment


#
# def __get_model(client, model_name, stage):
#     """Check if model is already stored locally at its latest version or download if necessary.
#         :rtype: (mlflow.pyfunc.PyFuncModel, string)
#     """




