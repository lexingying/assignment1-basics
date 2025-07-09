import json

with open("cs336_basics/results/owt/vocab.json", 'r') as f:
    vocab = json.load(f)

top_10_tokens = sorted(vocab.items(), key=lambda item: len(item[1]), reverse=True)[:10]

decoded_tokens = []
for token_key, token_value in top_10_tokens:
    decoded_token = bytes(token_value).decode('utf-8', errors='replace')
    decoded_tokens.append(decoded_token)

print("Top 10 tokens by length:")
for token_key, decoded_token in zip(top_10_tokens, decoded_tokens):
    print(f"Token ID: {token_key}, Decoded Token: {decoded_token}")