import tensorflow as tf
import zmq
import json


def process():
    server = Server()
    server.eventLoop()


class Server:
    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:8888")
        self.active = True
        print("Server Started")

    def eventLoop(self):
        while self.active:
            message = self.socket.recv_json()
            response = self.handleMessage(message)
            self.socket.send_json(response)

    def loadModel(self, modelFile):
        print("Loading Model " + modelFile)
        self.model = tf.keras.models.load_model(modelFile, compile=False)

    def createResponse(self, opCode, payload):
        response = {
            "opCode": opCode,
            "payload": payload
        }
        return response

    def handleMessage(self, message):
        response = {
            "loadModel": self.loadModelMessage,
            "shutdown": self.shutdownMessage
        }[message["opCode"]](message)

        return response

    def loadModelMessage(self, message):
        self.loadModel(message["payload"])
        response = self.createResponse("loadModelResponse", "success")
        return response

    def shutdownMessage(self, message):
        self.active = False
        print("Shutting Down Server")
        response = self.createResponse("shutdownResponse", "success")
        return response
