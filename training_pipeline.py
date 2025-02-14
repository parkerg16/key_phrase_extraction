import os
import glob
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Download the necessary NLTK tokenizers
nltk.download('punkt')
nltk.download('punkt_tab')
# Configuration
# -------------------------------
# Folder containing chapter text files produced by your chunking.py
chunks_folder = "book_chunks"  # adjust if your folder is named differently

# Where to save the fine-tuned model
output_model_path = "fine_tuned_model"

# Training parameters
batch_size = 256
num_epochs = 10  # Adjust this based on your data size and desired training time

# -------------------------------
# Step 1: Build Training Examples from Chapters
# -------------------------------
# We'll read each chunk, split into sentences, and for unsupervised training we create a pair
# by using the same sentence twice (letting dropout introduce variation).
training_examples = []

# Glob all .txt files in the chunks folder
chapter_files = glob.glob(os.path.join(chunks_folder, "*.txt"))
print(f"Found {len(chapter_files)} chapter files.")

for file_path in chapter_files:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split the chapter text into sentences
    sentences = sent_tokenize(text)
    for sentence in sentences:
        sentence = sentence.strip()
        # Optionally filter out very short sentences
        if len(sentence) < 30:
            continue
        # Create an InputExample with the sentence duplicated.
        # During training, dropout will give slightly different representations.
        training_examples.append(InputExample(texts=[sentence, sentence]))

print(f"Collected {len(training_examples)} training examples from your chapters.")

# -------------------------------
# Step 2: Load and Prepare the Pre-trained Model
# -------------------------------
# You can choose a base model; here we use 'all-MiniLM-L6-v2' as an example.
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Move the model to GPU if available
if model.device.type != "cuda":
    model = model.to("cuda")
    print("Model moved to CUDA.")

# -------------------------------
# Step 3: Create DataLoader and Loss Function
# -------------------------------
train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
# Use MultipleNegativesRankingLoss for unsupervised contrastive training.
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# Calculate warmup steps (e.g. 10% of total steps)
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
print(f"Warmup steps: {warmup_steps}")

# -------------------------------
# Step 4: Fine-tune the Model
# -------------------------------
print("Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_model_path,
    show_progress_bar=True
)
print(f"Fine-tuning complete. Model saved to {output_model_path}")
