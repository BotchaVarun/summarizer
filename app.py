import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
# Import your specific function
from pdf_summarizer import summarize_pdf

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # 1. Save file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 2. Get settings from Frontend
        top_k = int(request.form.get('top_k', 6))
        diversity = float(request.form.get('diversity', 0.65))
        use_torch = request.form.get('use_torch') == 'true'

        # 3. Run YOUR summarizer code
        result = summarize_pdf(
            pdf_path=filepath,
            top_k=top_k,
            diversity_lambda=diversity,
            use_torch_reranker=use_torch,
            verbose=False # Keep logs clean
        )

        # 4. Format data for JSON response
        response_data = {
            "sentences": [
                {
                    "text": s.text,
                    "score": round(s.final_score, 4),
                    "cluster_id": s.section_index # Mapping section to cluster color
                } 
                for s in result.metadata
            ],
            "stats": {
                "original_sentences": result.total_sentences,
                "original_words": result.total_words,
                "compression_ratio": round(result.compression_ratio * 100, 1)
            }
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    finally:
        # 5. Clean up (delete temp file)
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)