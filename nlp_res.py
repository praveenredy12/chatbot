from flask import Flask
from flask_restful import Resource, Api
from chatbot import ChatBot

app = Flask(__name__)
api = Api(app)


class Response(Resource):
    def get(self,text):
        return ChatBot(text).get_response()


api.add_resource(Response, '/res/<text>')


if __name__ == '__main__':
    app.run()
