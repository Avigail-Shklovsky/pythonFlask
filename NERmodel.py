from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-joint')
model = AutoModel.from_pretrained('dicta-il/dictabert-joint', trust_remote_code=True)

model.eval()

def analyze_text_with_model(sentence: str):
    return model.predict([sentence], tokenizer, output_style='json')


 