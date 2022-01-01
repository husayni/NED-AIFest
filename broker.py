import os

import numpy as np
from celery import Celery, Task
from celery.utils.log import get_task_logger
from keras.preprocessing import image
from tensorflow.keras.models import load_model

BROKER_URI = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
BACKEND_URI = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

client = Celery(__name__)
client.conf.task_serializer = "pickle"
client.conf.result_serializer = "pickle"
client.conf.accept_content = ["pickle"]
client.conf.broker_url = BROKER_URI
client.conf.result_backend = BACKEND_URI

logger = get_task_logger(__name__)


class PredictTask(Task):
    abstract = True

    def __init__(self):
        super().__init__()
        self.emotions = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        ]
        self.model = None

    def __call__(self, *args, **kwargs):
        if not self.model:
            logger.info("Loading Model...")
            self.model = load_model(r"weights\EmotionDetection_custom.h5")
            logger.info("Model loaded")
        return self.run(*args, **kwargs)


@client.task(base=PredictTask, bind=True)
def process_job(self, filepath):
    img = image.load_img(filepath, target_size=(128, 128, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    pred = self.model.predict(x)
    os.remove(filepath)
    return self.emotions[np.argmax(pred[0]) - 1]
