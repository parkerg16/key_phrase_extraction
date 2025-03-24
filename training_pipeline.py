import os
import glob
import nltk
import torch
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Download the necessary NLTK tokenizers
nltk.download('punkt')
nltk.download('punkt_tab')

# Configuration
chunks_folder = 'Chapters'  # Folder containing chapter text files
output_model_path = "fine_tuned_model"
batch_size = 2 # This is model dependent I am using 16 on my 4090 using the larger model, but can approach 100 on smaller models
num_epochs = 1  # Adjust based on data size and desired training time this can take a really long time especially on older machines so be careful


# Step 1: Build Training and Testing Examples from Chapters

# Get a sorted list of all .txt files in the chunks folder
chapter_files = sorted(glob.glob(os.path.join(chunks_folder, "*.txt")))
total_chapters = len(chapter_files)
print(f"Found {total_chapters} chapter files.")

# Split into training and testing sets: first 15 for training, last 4 for testing
train_chapters = chapter_files[:15]
test_chapters = chapter_files[15:19]

training_examples = []
testing_examples = []


def create_examples(file_list, example_list, set_name=""):
    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Split the chapter text into sentences
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sentence = sentence.strip()
            # Optionally filter out very short sentences
            if len(sentence) < 30:
                continue
            # Create an InputExample with the sentence duplicated for unsupervised training
            example_list.append(InputExample(texts=[sentence, sentence]))
    print(f"Collected {len(example_list)} {set_name} examples.")


create_examples(train_chapters, training_examples, set_name="training")
create_examples(test_chapters, testing_examples, set_name="testing")


# Step 2: Load and Prepare the Pre-trained Model

model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("GPU not detected, using CPU instead.")
model.to(device)


# Step 3: Create DataLoader and Loss Function for Training

train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
train_loss = losses.MultipleNegativesRankingLoss(model=model)

warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
print(f"Warmup steps: {warmup_steps}")


# Step 4: Fine-tune the Model

print("Starting fine-tuning...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path=output_model_path,
    show_progress_bar=True
)
print(f"Fine-tuning complete. Model saved to {output_model_path}")

