import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
from collections import Counter
import json

# Assuming these are defined in separate files
from positional_encoding import PositionalEncoding
from transformer import build_transformer
from input_embedding import InputEmbedding

# Create directory to save model weights
os.makedirs('model_weights', exist_ok=True)

def save_sampled_data(data, filename="sampled_train_data.json"):
    # Save the sampled data to a JSON file for reproducibility
    serializable_data = [{"de": example["translation"]["de"], "en": example["translation"]["en"]} for example in data]
    with open(filename, 'w') as f:
        json.dump(serializable_data, f, ensure_ascii=False, indent=4)

def load_and_preprocess_data(batch_size=4, max_length=128, sample_size=100000):
    # Load the full dataset
    dataset = load_dataset("wmt14", "de-en")

    # Sample the dataset
    random.seed(42)
    sampled_train = dataset['train'].shuffle(seed=42).select(range(sample_size))
    sampled_val = dataset['train'].shuffle(seed=42).select(range(sample_size))
    save_sampled_data(sampled_train)

    # Initialize tokenizers
    src_tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=max_length)
    tgt_tokenizer = AutoTokenizer.from_pretrained("t5-small", model_max_length=max_length)

    # Ensure special tokens are set
    special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
    src_tokenizer.add_special_tokens(special_tokens)
    tgt_tokenizer.add_special_tokens(special_tokens)

    def preprocess_function(examples):
        # Preprocess function to tokenize the source and target texts
        src_texts = [example['de'] for example in examples['translation']]
        tgt_texts = [example['en'] for example in examples['translation']]

        model_inputs = src_tokenizer(src_texts, max_length=max_length, truncation=True, padding='max_length')
        labels = tgt_tokenizer(tgt_texts, max_length=max_length, truncation=True, padding='max_length')

        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    # Preprocess the datasets
    train_ds = sampled_train.map(preprocess_function, batched=True, remove_columns=sampled_train.column_names)
    val_ds = sampled_val.map(preprocess_function, batched=True, remove_columns=sampled_val.column_names)

    # Set the tensor format
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader, src_tokenizer, tgt_tokenizer

def generate_causal_mask(seq_len):
    # Generate a causal mask for the decoder to prevent attending to future tokens
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def get_model(src_vocab_size, tgt_vocab_size, src_seq_length=350, tgt_seq_length=350):
    # Build the transformer model
    model = build_transformer(src_vocab_size, tgt_vocab_size, 512, 8, 6, src_seq_length=src_seq_length, tgt_seq_length=tgt_seq_length)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def analyze_token_distribution(loader, tokenizer):
    # Analyze token distribution in the dataset
    token_counts = Counter()
    for batch in loader:
        token_counts.update(batch['labels'].flatten().tolist())

    print("Token distribution:")
    for token, count in token_counts.most_common(30):
        print(f"{tokenizer.decode([token])}: {count}")

def custom_loss(outputs, targets, pad_token_id, smoothing=0.1):
    # Custom loss function with optional label smoothing
    vocab_size = outputs.size(-1)
    outputs = outputs.contiguous().view(-1, vocab_size)
    targets = targets.contiguous().view(-1)

    non_pad_mask = (targets != pad_token_id)
    n_tokens = non_pad_mask.sum()

    # Apply label smoothing
    if smoothing > 0:
        smoothed_targets = torch.full_like(outputs, smoothing / (vocab_size - 2))
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
        smoothed_targets[:, pad_token_id] = 0
        smoothed_targets.masked_fill_((targets == pad_token_id).unsqueeze(1), 0)
    else:
        smoothed_targets = torch.zeros_like(outputs)
        smoothed_targets.scatter_(1, targets.unsqueeze(1), 1)

    loss = -smoothed_targets * outputs.log_softmax(dim=-1)
    loss = loss.sum(dim=-1)
    loss = loss.masked_select(non_pad_mask).sum() / n_tokens
    return loss

def train_model():
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, src_tokenizer, tgt_tokenizer = load_and_preprocess_data(sample_size=4)

    src_vocab_size = len(src_tokenizer)
    tgt_vocab_size = len(tgt_tokenizer)

    model = get_model(src_vocab_size, tgt_vocab_size)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00007, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    # Analyze token distribution
    print("Analyzing token distribution in training data:")
    analyze_token_distribution(train_loader, tgt_tokenizer)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch+1}/{num_epochs}")

        for batch in batch_iterator:
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = (src != src_tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
            tgt_mask = generate_causal_mask(tgt_input.size(1)).to(device)

            try:
                encoded = model.encode(src, src_mask)
                decoded = model.decode(tgt_input, encoded, src_mask, tgt_mask)
                outputs = model.project(decoded)

                # Apply mask to output logits
                pad_mask = (tgt_output != tgt_tokenizer.pad_token_id).float()
                outputs = outputs + (1.0 - pad_mask.unsqueeze(-1)) * -1e9

                loss = custom_loss(outputs, tgt_output, tgt_tokenizer.pad_token_id)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1

                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            except RuntimeError as e:
                print(f"Error in batch: {e}")
                continue

        avg_train_loss = total_loss / len(train_loader)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['input_ids'].to(device)
                tgt = batch['labels'].to(device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask = (src != src_tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)
                tgt_mask = generate_causal_mask(tgt_input.size(1)).to(device)

                try:
                    encoded = model.encode(src, src_mask)
                    decoded = model.decode(tgt_input, encoded, src_mask, tgt_mask)
                    outputs = model.project(decoded)

                    # Apply mask to output logits
                    pad_mask = (tgt_output != tgt_tokenizer.pad_token_id).float()
                    outputs = outputs + (1.0 - pad_mask.unsqueeze(-1)) * -1e9

                    loss = custom_loss(outputs, tgt_output, tgt_tokenizer.pad_token_id)
                    total_val_loss += loss.item()
                except RuntimeError as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Update learning rate
        scheduler.step(avg_val_loss)

        # Evaluate model performance (you may want to implement this separately)
        if (epoch + 1) % 5 == 0:
            evaluate_model(model, val_loader, src_tokenizer, tgt_tokenizer, device)

    torch.save(model.state_dict(), 'model_weights/best_model.pth')

    print("Training completed.")

def check_output_encoding(model, src, src_mask, tgt_tokenizer, device):
    # Function to inspect the output encoding of the model
    model.eval()
    with torch.no_grad():
        encoded = model.encode(src, src_mask)

        # Generate first token
        tgt_input = torch.full((src.size(0), 1), tgt_tokenizer.bos_token_id, device=device)
        tgt_mask = generate_causal_mask(1).to(device)
        decoded = model.decode(tgt_input, encoded, src_mask, tgt_mask)
        logits = model.project(decoded[:, -1])

        # Print raw output probabilities
        print("Raw model output probabilities:")
        print(logits)

        # Print top 5 token probabilities
        top_probs, top_indices = torch.topk(logits, k=5, dim=-1)
        print("\nTop 5 tokens and their log probabilities:")
        for prob, idx in zip(top_probs[0], top_indices[0]):
            token = tgt_tokenizer.decode([idx])
            print(f"{token}: {prob.item():.4f}")

# Add this to your evaluate_model function:

def evaluate_model(model, val_loader, src_tokenizer, tgt_tokenizer, device):
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)

            src_mask = (src != src_tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

            encoded = model.encode(src, src_mask)

            # Generate translations
            max_len = 128
            translations = torch.full((src.size(0), max_len), tgt_tokenizer.pad_token_id, device=device)
            translations[:, 0] = tgt_tokenizer.bos_token_id

            for i in range(1, max_len):
                tgt_mask = generate_causal_mask(i).to(device)
                decoded = model.decode(translations[:, :i], encoded, src_mask, tgt_mask)
                logits = model.project(decoded[:, -1])

                # Debug: Print top 5 token probabilities for each step
                # top_probs, top_indices = torch.topk(logits, k=5, dim=-1)
                # print(f"\nStep {i}:")
                # for prob, idx in zip(top_probs[0], top_indices[0]):
                #     token = tgt_tokenizer.decode([idx])
                #     print(f"{token}: {prob.item():.4f}")

                next_token = torch.argmax(logits, dim=-1)
                translations[:, i] = next_token

                if (next_token == tgt_tokenizer.eos_token_id).all():
                    break

            # Print examples
            for i in range(min(3, src.size(0))):
                print("\nExample", i+1)
                print("Source:", src_tokenizer.decode(src[i], skip_special_tokens=True))
                print("Target:", tgt_tokenizer.decode(tgt[i], skip_special_tokens=True))
                print("Predicted:", tgt_tokenizer.decode(translations[i], skip_special_tokens=True))

            break  # Only process one batch for demonstration

    # Add vocabulary inspection
    # print("\nTarget Vocabulary Inspection:")
    # inspect_vocabulary(tgt_tokenizer)
    # check_output_encoding(model, src[:1], src_mask[:1], tgt_tokenizer, device)

def inspect_vocabulary(tokenizer, n=10):
    # Inspect the vocabulary of the tokenizer
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"\nFirst {n} tokens:")
    for i in range(min(n, len(tokenizer))):
        print(f"{i}: {tokenizer.decode([i])}")

    print(f"\nLast {n} tokens:")
    for i in range(max(0, len(tokenizer)-n), len(tokenizer)):
        print(f"{i}: {tokenizer.decode([i])}")

    # Check for specific tokens
    special_tokens = ['<unk>', '<pad>', '<bos>', '<eos>']
    print("\nSpecial tokens:")
    for token in special_tokens:
        if token in tokenizer.vocab:
            print(f"{token}: {tokenizer.vocab[token]}")
        else:
            print(f"{token}: Not found")

    # Check for some common English words
    common_words = ['the', 'a', 'is', 'to', 'and', 'in', 'it', 'that', 'of', 'for']
    print("\nCommon English words:")
    for word in common_words:
        if word in tokenizer.vocab:
            print(f"{word}: {tokenizer.vocab[word]}")
        else:
            print(f"{word}: Not found")

# Call the function to start training
if __name__ == "__main__":
    train_model()
