from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

app = FastAPI(
    title="Fake News Detection API",
    description="REST API serving a BERT model trained on GossipCop & PolitiFact data."
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load tokenizer and YOUR trained model checkpoint
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('./results_fakenewsnet/checkpoint-1160')
model.to(device)
model.eval()

# 2. Define the expected input from the user
class NewsRequest(BaseModel):
    text: str

# 3. Create the prediction endpoint
@app.post("/predict")
def classify_news(request: NewsRequest):
    # Tokenize exactly like you did in the notebook
    encodings = tokenizer(
        request.text,
        add_special_tokens=True,
        max_length=128, 
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(output.logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=1)
    
    # In your code: 0 = True, 1 = Fake
    label = "Fake News" if predicted_class.item() == 1 else "True News"
    
    return {
        "prediction": label,
        "confidence": f"{round(confidence.item() * 100, 2)}%"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)