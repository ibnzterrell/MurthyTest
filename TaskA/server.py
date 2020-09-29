import tensorflow as tf
import zmq
import json
import numpy as np


def inferenceProcess():
    inferenceServer = InferenceServer()
    inferenceServer.eventLoop()


class InferenceServer:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.portBinding = "tcp://*:8888"
        self.socket.bind(self.portBinding)
        self.active = True
        print("Inference Server Started on " + self.portBinding)

    def eventLoop(self):
        while self.active:
            message = self.socket.recv_json()
            response = self.handleMessage(message)
            self.socket.send_json(response)

    def createResponse(self, opCode, payload):
        response = {
            "opCode": opCode,
            "payload": payload
        }
        return response

    def handleMessage(self, message):
        response = {
            "loadModel": self.loadModelMessageHandler,
            "predict": self.predictMessageHandler,
            "shutdown": self.shutdownMessageHandler
        }[message["opCode"]](message)

        return response

    def loadModelMessageHandler(self, message):
        modelFile = message["payload"]
        print("Loading Model " + modelFile)
        self.model = tf.keras.models.load_model(modelFile, compile=False)
        response = self.createResponse("loadModelResponse", "success")
        return response

    def predictMessageHandler(self, message):
        img = message["payload"]
        X = np.expand_dims(img, axis=0).astype("float32") / 255.
        X = tf.image.resize(X, size=[512, 512])

        Y = self.model.predict(X)
        response = self.createResponse("predictResponse", Y.tolist())
        return response

    def shutdownMessageHandler(self, message):
        self.active = False
        print("Shutting Down Inference Server")
        response = self.createResponse("shutdownResponse", "success")
        return response
