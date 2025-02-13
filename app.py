from builtins import Exception, int, print, str
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from callback import callback_to_nextjs  # Import the callback function
import uuid
import re


app = Flask(__name__)
cors = CORS(app, resources={r"/analyze": {"origins": ["http://127.0.0.1:8080","http://localhost:3000", "https://pythonflask-production.up.railway.app",re.compile(r"https://.*\.vercel\.app"),]}})


@app.route('/analyze', methods=['POST'])
def analyze_text():
    print("in the py func")
    from NERmodel import analyze_text_with_model 
    data = request.json
    sentence = data.get('text')
    vercelUrl = data.get('vercelUrl')

    if not sentence:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = analyze_text_with_model(sentence)
        print(sentence)

        #  Generate a unique job ID using uuid4
        job_id = str(uuid.uuid4())

         # Send the result to Next.js using the callback function
        callback_to_nextjs(job_id, result,vercelUrl) 


        return jsonify({"embeddings": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
   port = int(os.getenv("PORT", 8080))  # Default to 8000 if PORT is not set
   app.run(host="0.0.0.0", port=port)
