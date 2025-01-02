from transformers import AutoModel, AutoTokenizer

# Specify a cache directory to store model files on disk
cache_directory = "./cache"

# Load tokenizer and model with caching
tokenizer = AutoTokenizer.from_pretrained(
    'dicta-il/dictabert-joint',
    cache_dir=cache_directory  # Specify cache directory
)
model = AutoModel.from_pretrained(
    'dicta-il/dictabert-joint',
    cache_dir=cache_directory,  # Specify cache directory
    trust_remote_code=True
)

model.eval()

def analyze_text_with_model(sentence: str):
    # Use the same prediction logic as localhost
    return model.predict([sentence], tokenizer, output_style='json')


