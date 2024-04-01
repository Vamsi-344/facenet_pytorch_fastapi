from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from config import CONFIG
from inception_resnet_v1 import InceptionResnetV1
from mtcnn import MTCNN
import os
import sys
import torch
import uvicorn

app = FastAPI(title="FaceNet API", description="Deploying the facenet_pytorch python library 'MTCNN' and 'FaceNet' models using FastAPI")

app.add_middleware(CORSMiddleware, allow_origins=["*"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model
    model = InceptionResnetV1(pretrained='vggface2').eval()
    mtcnn = MTCNN()

     # add model and other preprocess tools too app state
    app.package = {
        "mtcnn": mtcnn,
        "model": model
    }


@app.get("/", description="Welcome to the FaceNet FastAPI")
def root():
    return {"message": "Hello there!, This is facenet_pytorch FastAPI"}

# @app.post('/api/v1/predict')
# def do_predict(request: Request, body: InferenceInput):
#     """
#     Perform prediction on input data
#     """

#     logger.info('API predict called')
#     logger.info(f'input: {body}')

#     # prepare input data
#     X = [body.sepal_length, body.sepal_width,
#          body.petal_length, body.petal_width]

#     # run model inference
#     y = predict(app.package, [X])[0]

#     # generate prediction based on probablity
#     pred = ['setosa', 'versicolor', 'virginica'][y.argmax()]

#     # round probablities for json
#     y = y.tolist()
#     y = list(map(lambda v: round(v, ndigits=CONFIG['ROUND_DIGIT']), y))

#     # prepare json for returning
#     results = {
#         'setosa': y[0],
#         'versicolor': y[1],
#         'virginica': y[2],
#         'pred': pred
#     }

#     logger.info(f'results: {results}')

#     return {
#         "error": False,
#         "results": results
#     }

@app.get("/about")
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)