from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pathlib import Path
import os
import argparse

def main(args):
    # Load model (fine-tuned or base)
    if os.path.isdir(args.model_path):
        print(f"Loading fine-tuned model from: {args.model_path}")
        model = SentenceTransformer(model_name_or_path=args.model_path)
    else:
        # Initialize KeyBert with a standard pre-trained model
        # Options include: 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'distilbert-base-nli-mean-tokens', 'distilroberta-base', 'paraphrase-multilingual-MiniLM-L12-v2'
        model = SentenceTransformer(model_name_or_path=args.model_path)

    # 1. Define training examples
    train_examples = [
        InputExample(texts=["deep learning", "neural networks"], label=0.9),
        InputExample(texts=["machine learning", "support vector machines"], label=0.8),
        InputExample(texts=["data mining", "basketball"], label=0.1),
        InputExample(texts=["k-means", "clustering"], label=0.95),
        InputExample(texts=["training set", "validation set"], label=0.7)
    ]

    # 3. Prepare data loader + loss
    batch_size = 4
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    # 4. Define where to save your model
    epochs = 3
    warmup_steps = 10
    output_path = Path(args.output_path or f"models/keybert/{args.option}_finetuned_model")
    os.makedirs(output_path, exist_ok=True)

    # 5. Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        output_path=output_path
    )
    print(f"Fine-tuned model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train from base or existing model")
    parser.add_argument("--option", type=str, default="existing", choices=["existing", "base"], help="Run on 'existing' or 'base' model")
    parser.add_argument("--model_path", type=str, default="", help="Path to existing model to load")
    parser.add_argument("--output_path", type=str, default="", help="Optional: Save model to this path")
    args = parser.parse_args()
    main(args)