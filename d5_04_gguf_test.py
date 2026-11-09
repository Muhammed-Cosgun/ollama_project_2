
from pathlib import Path
from gguf import GGUFReader

model_path = (
    Path.home()
    / ".ollama"
    / "models"
    / "blobs"
    / "sha256-0800cbac9c2064dde519420e75e512a83cb360de3ad5df176185dc69652fc515"
)

reader = GGUFReader(str(model_path))

# Tokenizer türünü oku
for alan in ["tokenizer.ggml.model", "tokenizer.ggml.pre"]:
    field = reader.get_field(alan)

    if field is not None:
        deger = field.contents()

        if isinstance(deger, bytes):
            deger = deger.decode("utf-8")

        print(f"{alan}: {deger}")

# Token sözlüğünün büyüklüğünü göster
field = reader.get_field("tokenizer.ggml.tokens")

if field is not None:
    print("Token sözlüğü boyutu:", len(field.data))
