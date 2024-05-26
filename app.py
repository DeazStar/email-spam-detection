from flask import Flask, request, jsonify
from starter import predict_naive_bayes

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_spam_detection():
    data = request.json
    
    if 'email' in data:
        email = data['email']
        
        result = predict_naive_bayes(email)
        
        response = {
            'status': 'success',
            'message': f"The Email is {result}"
        }
        
        return jsonify(response), 200
    else:
        error_response = {
            'status': 'error',
            'message': '`email field is required',
        }
        
        return jsonify(error_response), 400
if __name__ == '__main__':
    app.run()