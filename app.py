from flask import Flask, request, jsonify
from flask_cors import CORS
from NERmodel import analyze_text_with_model 

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    sentence = data.get('text')

    # Ensure sentence is not empty or None
    if not sentence:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = analyze_text_with_model(sentence)
        return jsonify({"embeddings": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
