from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import get_hyde_response, get_normal_retriever_response

app = Flask(__name__)
CORS(app)

@app.route('/hyde_query', methods=['POST'])
def hyde_query():
    query = request.form.get('query')
    response = jsonify({'response': get_hyde_response(query)})
    return response


@app.route('/normal_retrieval_query', methods=['POST'])
def normal_retrieval_query():
    query = request.form.get('query')
    response = jsonify({'response': get_normal_retriever_response(query)})
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')