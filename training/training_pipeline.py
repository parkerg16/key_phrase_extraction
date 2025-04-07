from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

# 1. Define training examples (mocked for now, replace with your own)
train_examples = [
    InputExample(texts=["deep learning", "neural networks"], label=0.9),
    InputExample(texts=["machine learning", "support vector machines"], label=0.8),
    InputExample(texts=["data mining", "basketball"], label=0.1),
    InputExample(texts=["k-means", "clustering"], label=0.95),
    InputExample(texts=["training set", "validation set"], label=0.7)
]

# 2. Load base transformer
model_name = "distilroberta-base-msmarco-v2"
model = SentenceTransformer(model_name)

# 3. Prepare data loader + loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
train_loss = losses.CosineSimilarityLoss(model=model)

# 4. Define where to save your model
output_path = "models/keybert/my_finetuned_model"
os.makedirs(output_path, exist_ok=True)

# 5. Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
    show_progress_bar=True,
    output_path=output_path
)

print(f"✅ Fine-tuned model saved to: {output_path}")
