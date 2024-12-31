# from builtins import Exception, int, str
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os

# app = Flask(__name__)
# CORS(app)

# @app.route('/analyze', methods=['POST'])
# def analyze_text():
#     from NERmodel import analyze_text_with_model  # Import here to save memory
#     data = request.json
#     sentence = data.get('text')

#     if not sentence:
#         return jsonify({"error": "No text provided"}), 400

#     try:
#         result = analyze_text_with_model(sentence)
#         return jsonify({"embeddings": result})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=True)


from builtins import Exception, int, print, str
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


@app.route('/analyze', methods=['POST'])
def analyze_text():
    print("in the py func")
    from NERmodel import analyze_text_with_model  # Import here to save memory
    data = request.json
    sentence = data.get('text')

    if not sentence:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = analyze_text_with_model(sentence)
        return jsonify({"embeddings": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=8080, debug=True)
