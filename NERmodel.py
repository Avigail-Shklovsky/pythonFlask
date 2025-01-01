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



from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

# Specify a cache directory to store model files on disk
cache_directory = "./cache"

# Load tokenizer and model with caching
tokenizer = AutoTokenizer.from_pretrained(
    'dicta-il/dictabert-joint',
    cache_dir=cache_directory  # Specify cache directory
)
model = AutoModelForTokenClassification.from_pretrained(
    'dicta-il/dictabert-joint',
    cache_dir=cache_directory,  # Specify cache directory
    trust_remote_code=True
)

model.eval()

def analyze_text_with_model(sentence: str):
    print(sentence)
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Run the model on the tokenized inputs
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predictions (e.g., labels or logits)
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Convert predicted token IDs to tokens
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())
    
    return predicted_tokens



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

