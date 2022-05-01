import os
import io
import json
import requests
import logging
import numpy as np
import pickle
import pandas as pd
from text_vectorian import SentencePieceVectorian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
vectorian = SentencePieceVectorian()
input_len = 64
dim = 100

def handler(data, context):
    """Handle request.
    Args:
        data (obj): the request data
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, (optional) response content type
    """
    processed_input = _process_input(data, context)
    response = requests.post(context.rest_uri, data=processed_input)

    return _process_output(response, context)


def _process_input(data, context):
    if context.request_content_type == 'application/json':
        body = data.read().decode('utf-8')

        param = json.loads(body)
        query = param['q']
        features = np.zeros((1, input_len))
        inputs = vectorian.fit(query).indices

        for i, index in enumerate(inputs):
            if i >= input_len:
                break
            pos = input_len - len(inputs) + i
            features[0, pos] = index
    
        return json.dumps({
            'inputs': features.tolist()
        })

    raise ValueError('{{"error": "unsupported content type {}"}}'.format(
        context.request_content_type or "unknown"))


def _process_output(data, context):
    if data.status_code != 200:
        raise ValueError(data.content.decode('utf-8'))

    response_content_type = 'application/json'

    body = json.loads(data.content.decode('utf-8'))
    predicts = body['outputs'][0]

    labels_path = '/opt/ml/model/code/labels.pickle'

    with open(labels_path, mode='rb') as f:
        labels = pickle.load(f)
    rets = _create_response(predicts, labels)

    logger.warn(rets)

    return json.dumps(rets), response_content_type

def _create_response(predicts, labels):
    rets = []

    for index in np.argsort(predicts)[::-1]:
        label = labels['index2label'][index]
        prob = predicts[index]
        rets.append({
            'label': label,
            'prob': prob
        })

    return rets
