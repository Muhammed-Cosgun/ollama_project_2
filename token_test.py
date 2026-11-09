
import subprocess
import ast
from pathlib import Path

# 1. llama-tokenize.exe dosyasının konumu
exe = Path(
    r"C:\Users\Muhammed\Desktop\llama_cpp"
    r"\llama-b12202-bin-win-cpu-x64\llama-tokenize.exe"
)

# 2. Yerel EmbeddingGemma modelinin konumu
model = (
    Path.home()
    / ".ollama"
    / "models"
    / "blobs"
    / "sha256-0800cbac9c2064dde519420e75e512a83cb360de3ad5df176185dc69652fc515"
)

# 3. Tokenlara ayıracağımız metin
metin = "TCKN kaç haneli olmalıdır?"

# 4. Token kimliklerini elde et
sonuc = subprocess.run(
    [
        str(exe),
        "-m", str(model),
        "-p", metin,
        "--ids",
        "--no-bos",
        "--offline",
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
    check=True,
)

tokenlar = ast.literal_eval(sonuc.stdout.strip())

# 5. Tokenların metinsel gösterimini elde et
sonuc_metin = subprocess.run(
    [
        str(exe),
        "-m", str(model),
        "-p", metin,
        "--no-bos",
        "--offline",
    ],
    capture_output=True,
    text=True,
    encoding="utf-8",
    check=True,
)

# 6. Sonuçları ekrana yazdır
print("Metin:", metin)

print("\nToken kimlikleri:")
print(tokenlar)

print("\nToplam token sayısı:", len(tokenlar))

print("\nTokenların metinsel gösterimi:")
print(sonuc_metin.stdout)