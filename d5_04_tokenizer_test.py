from transformers import AutoTokenizer

# EmbeddingGemma modelinin tokenizer'ını yükle
tokenizer = AutoTokenizer.from_pretrained(
    "google/embeddinggemma-300m"
)

# Örnek metin
metin = "Alıcı kimlik bilgisinin türü TCKN ise bu değer 11 haneden oluşmalıdır."

# Metni tokenlara dönüştür
tokenlar = tokenizer.encode(
    metin,
    add_special_tokens=False
)

print("Tokenizer:", tokenizer.name_or_path)
print("Metin:", metin)
print("Token sayısı:", len(tokenlar))
print("Token ID'leri:", tokenlar)

print("\n--- TOKENLAR ---")
for i, token_id in enumerate(tokenlar, 1):
    print(i, tokenizer.convert_ids_to_tokens(token_id))
