import torch
from transformers import AutoTokenizer
from transformer import build_transformer
import torch.nn.functional as F

# Load tokenizers
 # Initialize tokenizers
src_tokenizer = AutoTokenizer.from_pretrained("t5-small")
tgt_tokenizer = AutoTokenizer.from_pretrained("t5-small")


# Ensure special tokens are set
special_tokens = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>'}
src_tokenizer.add_special_tokens(special_tokens)
tgt_tokenizer.add_special_tokens(special_tokens)

def load_model(model_path, src_vocab_size, tgt_vocab_size):
    model = build_transformer(src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, layers=6,
                              src_seq_length=350, tgt_seq_length=350)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def translate(model, src_text, max_length=128):
    model.eval()
    src = src_tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    src_mask = (src['input_ids'] != src_tokenizer.pad_token_id).unsqueeze(-2)

    print(f"Source tokens: {src_tokenizer.convert_ids_to_tokens(src['input_ids'][0])}")

    with torch.no_grad():
        memory = model.encode(src['input_ids'], src_mask)
        ys = torch.ones(1, 1).fill_(tgt_tokenizer.bos_token_id).type(torch.long)
        for i in range(max_length-1):
            tgt_mask = generate_square_subsequent_mask(ys.size(1)).type(torch.bool)
            out = model.decode(ys, memory, src_mask, tgt_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            print(f"Step {i+1}: Generated token: {tgt_tokenizer.convert_ids_to_tokens([next_word])[0]}")

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src['input_ids']).fill_(next_word)], dim=1)
            if next_word == tgt_tokenizer.eos_token_id:
                break

    ys = ys.view(-1).tolist()
    return tgt_tokenizer.decode(ys)

def check_model_output(model, src_text, max_length=128):
    model.eval()
    src = src_tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    src_mask = (src['input_ids'] != src_tokenizer.pad_token_id).unsqueeze(-2)

    with torch.no_grad():
        memory = model.encode(src['input_ids'], src_mask)
        ys = torch.ones(1, 1).fill_(tgt_tokenizer.bos_token_id).type(torch.long)
        out = model.decode(ys, memory, src_mask, None)
        prob = model.project(out[:, -1])
        print("Raw model output probabilities:")
        print(prob)
        print("Top 5 tokens:")
        top_5 = torch.topk(prob, 5)
        for i in range(5):
            token_id = top_5.indices[0][i].item()
            token = tgt_tokenizer.convert_ids_to_tokens([token_id])[0]
            print(f"{token}: {top_5.values[0][i].item():.4f}")

def check_model_parameters(model):
    print("Number of model parameters:", sum(p.numel() for p in model.parameters()))
    print("Sample of model parameters:")
    for name, param in list(model.named_parameters())[:5]:
        print(f"{name}: {param.data.mean().item():.4f}")

if __name__ == "__main__":
    # Load the trained model
    model_path = 'model_weights/best_model.pth'
    src_vocab_size = len(src_tokenizer)
    tgt_vocab_size = len(tgt_tokenizer)
    model = load_model(model_path, src_vocab_size, tgt_vocab_size)

    # Check model parameters
    check_model_parameters(model)

    # Example usage
    german_text = "Ich liebe maschinelles Lernen."
    print(f"\nInput (German): {german_text}")

    english_translation = translate(model, german_text)
    print(f"Translation (English): {english_translation}")

    # Check raw model output
    print("\nChecking raw model output:")
    check_model_output(model, german_text)

    # Interactive mode
    print("\nEnter German text to translate (or 'q' to quit):")
    while True:
        user_input = input("> ")
        if user_input.lower() == 'q':
            break
        translation = translate(model, user_input)
        print(f"Translation: {translation}")
        check_model_output(model, user_input)

    print("Goodbye!")