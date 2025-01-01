# from builtins import str
# from transformers import AutoModel, AutoTokenizer

# # Specify a cache directory to store model files on disk
# cache_directory = "./cache"

# # Load tokenizer and model with caching
# tokenizer = AutoTokenizer.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory  # Specify cache directory
# )
# model = AutoModel.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory,  # Specify cache directory
#     trust_remote_code=True
# )

# model.eval()

# def analyze_text_with_model(sentence: str):
#     return model.predict([sentence], tokenizer, output_style='json')


# rail 200, array of x19
# from transformers import AutoModelForTokenClassification, AutoTokenizer
# import torch

# # Specify a cache directory to store model files on disk
# cache_directory = "./cache"

# # Load tokenizer and model with caching
# tokenizer = AutoTokenizer.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory  # Specify cache directory
# )
# model = AutoModelForTokenClassification.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory,  # Specify cache directory
#     trust_remote_code=True
# )

# model.eval()

# def analyze_text_with_model(sentence: str):
#     print(sentence)
#     # Tokenize the input sentence
#     inputs = tokenizer(sentence, return_tensors="pt")

#     # Run the model on the tokenized inputs
#     with torch.no_grad():
#         outputs = model(**inputs)

#     # Get the predictions (e.g., labels or logits)
#     logits = outputs.logits
#     predicted_ids = torch.argmax(logits, dim=-1)

#     # Convert predicted token IDs to tokens
#     predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())
    
#     return predicted_tokens


# in the local works, rail 502
# from transformers import AutoModel, AutoTokenizer

# # Specify a cache directory to store model files on disk
# cache_directory = "./cache"

# # Load tokenizer and model with caching
# tokenizer = AutoTokenizer.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory  # Specify cache directory
# )
# model = AutoModel.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory,  # Specify cache directory
#     trust_remote_code=True
# )

# model.eval()

# def analyze_text_with_model(sentence: str):
#     # Use the same prediction logic as localhost
#     return model.predict([sentence], tokenizer, output_style='json')


# returns 200 - with an array of enmaty rokens
# from transformers import AutoModelForTokenClassification, AutoTokenizer
# import torch

# # Specify a cache directory to store model files on disk
# cache_directory = "./cache"

# # Load tokenizer and model with caching
# tokenizer = AutoTokenizer.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory  # Specify cache directory
# )
# model = AutoModelForTokenClassification.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory,  # Specify cache directory
#     trust_remote_code=True
# )

# model.eval()

# def analyze_text_with_model(sentence: str):
#     print("Input sentence:", sentence)
    
#     # Tokenize the input sentence
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    
#     # Run the model on the tokenized inputs
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Extract logits and determine predicted labels
#     logits = outputs.logits
#     predicted_ids = torch.argmax(logits, dim=-1)

#     # Align tokens with predictions
#     tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
#     predictions = [model.config.id2label[p.item()] for p in predicted_ids[0]]

#     # Combine tokens with their predicted labels
#     result = [{"token": token, "label": label} for token, label in zip(tokens, predictions)]

#     return result

# # Example usage
# result = analyze_text_with_model("This is an example sentence.")
# print("Result:", result)


# 502
# from transformers import AutoModel, AutoTokenizer
# import torch

# # Specify a cache directory to store model files on disk
# cache_directory = "./cache"

# # Load tokenizer and model with caching
# tokenizer = AutoTokenizer.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory  # Specify cache directory
# )
# model = AutoModel.from_pretrained(
#     'dicta-il/dictabert-joint',
#     cache_dir=cache_directory,  # Specify cache directory
#     trust_remote_code=True
# )

# model.eval()

# def analyze_text_with_model(sentence: str):
#     print("Input sentence:", sentence)

#     # Tokenize the input sentence
#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

#     # Run the model on the tokenized inputs
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     # Check if the model has a `predict` method
#     if hasattr(model, "predict"):
#         # Use the model's predict method for custom processing
#         predictions = model.predict([sentence], tokenizer, output_style="json")
#         return predictions
    
#     # If `predict` is not available, manually process embeddings (model-specific logic needed)
#     embeddings = outputs.last_hidden_state  # Example placeholder for embeddings
#     return embeddings



from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import logging

# Enable logging
logging.basicConfig(level=logging.DEBUG)

cache_directory = "./cache"

# Log the start of model loading
logging.debug("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-joint', cache_dir=cache_directory)
model = AutoModelForTokenClassification.from_pretrained('dicta-il/dictabert-joint', cache_dir=cache_directory, trust_remote_code=True)

# Convert model to half precision to save memory
model = model.half()

model.eval()

def analyze_text_with_model(sentence: str):
    logging.debug(f"Tokenizing sentence: {sentence}")
    
    # Tokenization with truncation
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    logging.debug(f"Inputs: {inputs}")

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())
    
    # Log the model's predictions
    logging.debug(f"Predicted Tokens: {predicted_tokens}")

    return predicted_tokens
