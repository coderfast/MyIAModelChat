import torch

# Load checkpoint
checkpoint = torch.load('chat_model.pth', map_location='cpu', weights_only=False)

print("Model checkpoint loaded successfully.")
print("Keys in checkpoint:", list(checkpoint.keys()))

if 'tokenizer' in checkpoint:
    tokenizer = checkpoint['tokenizer']
    print("Tokenizer found in checkpoint, vocab_size:", tokenizer.vocab_size)
else:
    print("Tokenizer not found in checkpoint.")

# Check if it's a state_dict or full checkpoint
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

print("Embedding weight shape:", state_dict['embedding.weight'].shape)
print("LSTM weight shapes:")
for key in state_dict.keys():
    if 'lstm' in key:
        print(f"  {key}: {state_dict[key].shape}")
print("FC weight shape:", state_dict['fc.weight'].shape)

# Check if weights are not all zeros or random
embedding_mean = state_dict['embedding.weight'].mean().item()
embedding_std = state_dict['embedding.weight'].std().item()
print(f"Embedding weight mean: {embedding_mean:.6f}, std: {embedding_std:.6f}")

if abs(embedding_mean) > 0.01 or embedding_std < 0.01 or embedding_std > 1.0:
    print("Model appears to have been trained (weights not random).")
else:
    print("Model may not have been trained properly.")