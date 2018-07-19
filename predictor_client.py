"""Predict classification on provided text.

Send request to a tensorflow_model_server.

   tensorflow_model_server --port=9000 --model_base_path=$export_dir_base

Usage:
   
   predictor_client.py --text='some text'

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import inputs
import text_utils

from grpc.beta import implementations
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import prediction_service_pb2


tf.flags.DEFINE_string('server', 'localhost:9000',
                       'TensorflowService host:port')
tf.flags.DEFINE_string("text", None, "Text to predict label of")
tf.flags.DEFINE_string("signature_def", "proba",
                       "Stored signature key of method to call (proba|embedding)")
FLAGS = tf.flags.FLAGS


def Request(text):
    example = inputs.BuildTextExample(text_utils.TokenizeText(text))
    request = classification_pb2.ClassificationRequest()
    request.model_spec.name = 'default'
    request.model_spec.signature_name = FLAGS.signature_def
    request.input.example_list.examples.extend([example])
    return request


def main(_):
    if not FLAGS.text:
        raise ValueError("No --text provided")
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = Request(FLAGS.text)
    result = stub.Classify(request, 10.0)  # 10 secs timeout
    print(result)


if __name__ == '__main__':
    tf.app.run()
    
