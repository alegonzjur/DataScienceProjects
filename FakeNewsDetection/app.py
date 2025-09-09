from flask import Flask, render_template, request, jsonify
import os

# Due to a OpenMP error.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
app = Flask(__name__)

# Model settings.
MODEL_PATH = "./models/bert_base_train_dir/checkpoint-13810" 
BASE_MODEL_NAME = "bert-base-uncased"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variables for model & tokenizer.
model = None
tokenizer = None

def load_model():
    """Loading base model"""
    global model, tokenizer
    
    try:
        print("Loading tokenizer...")
        # Tokenizer from checkpoint first.
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except:
            # If not, use base model.
            print(f"Loading tokenizer from base model: {BASE_MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        
        print("Loading base model...")
        # Loading.
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=2,  
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        
        print("Loading adapters LoRA...")
        # Loading the model with adapters.
        model = PeftModel.from_pretrained(base_model, MODEL_PATH)
        
        # Optimizing inference function by merging adapters.
        model = model.merge_and_unload()
        
        model.to(device)
        model.eval()
        print("Model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error while loading the model: {e}")
        print("Possible solutions:")
        print("1. Verify that BASE_MODEL_NAME is correct")
        print("2. Make sure Peft is installed: pip install peft")
        print("3. Check that the model checkpoint contains adapter_config.json")
        return False

def predict_news(title):
    """Predict function"""
    if model is None or tokenizer is None:
        return None, "Model not available"
    
    try:
        # Tokenizing the title.
        inputs = tokenizer(
            title,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Moving tensors.
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction.
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        
        # 0 = Fake, 1 = Real
        label = "Real" if predicted_class == 1 else "Fake"
        
        return {
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "fake": round(predictions[0][0].item() * 100, 2),
                "real": round(predictions[0][1].item() * 100, 2)
            }
        }, None
        
    except Exception as e:
        return None, f"Precition error: {str(e)}"

# Home route
@app.route('/')
def home():
    """Main page."""
    return render_template('index.html')

# Predict route.
@app.route('/predict', methods=['POST'])
def predict():
    """Route to make predictions."""
    try:
        data = request.get_json()
        title = data.get('title', '').strip()
        
        if not title:
            return jsonify({'error': 'Please, insert a title.'}), 400
        
        result, error = predict_news(title)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health')
def health():
    """Model status route"""
    model_status = "OK" if model is not None else "Error"
    return jsonify({
        'status': 'OK',
        'model_status': model_status,
        'device': str(device)
    })

if __name__ == '__main__':
    # Verify that the path exists.
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model wasn't found on the directory {MODEL_PATH}")
        print("Make sure that the path is correct.")
        exit(1)
    
    # Loading the model when the app opens.
    if load_model():
        print(f"App launched on device: {device}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Model could not be loaded. Verify settings.")
        exit(1)