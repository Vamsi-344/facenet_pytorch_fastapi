#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from config import CONFIG
import numpy as np


def preprocess(package: dict, input: list) -> list:
    """
    Preprocess data before running with model, for example detecting and aligning the image we got

    :param package: dict from fastapi state including model and processing objects
    :param package: list of input to be preprocessed
    :return: list of preprocessed input
    """

    # scale the data based with scaler fit during training
    mtcnn = package['mtcnn']
    input = mtcnn(input)

    return input


def predict(package: dict, input: list) -> np.ndarray:
    """
    Run model and get result

    :param package: dict from fastapi state including model and processing objects
    :param package: list of input values
    :return: numpy array of model output
    """

    # process data
    X = preprocess(package, input)

    # run model
    model = package['model']
    with torch.no_grad():
        # convert input from list to Tensor
        X = torch.Tensor(X)

        # move tensor to device
        X = X.to(CONFIG['DEVICE'])

        # run model
        y_pred = model(X)

    # convert result to a numpy array on CPU
    y_pred = y_pred.cpu().numpy()

    return y_pred