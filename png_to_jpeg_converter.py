import os
from PIL import Image

# Klasör yolu
folder_path = r"C:\codes\segment-anything-2\video"

# Klasördeki tüm dosyalar arasında gezin
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # Sadece .png dosyalarını kontrol et
        png_file_path = os.path.join(folder_path, filename)
        
        # Aynı isimle jpeg uzantılı dosya adı oluştur
        jpeg_file_path = os.path.join(folder_path, filename.replace(".png", ".jpeg"))
        
        # Görüntüyü aç ve jpeg formatında kaydet
        with Image.open(png_file_path) as img:
            img.convert('RGB').save(jpeg_file_path, "JPEG")
        
        # Orijinal PNG dosyasını sil
        os.remove(png_file_path)
        print(f"{filename} -> {jpeg_file_path} (PNG dosyası silindi)")

print("Tüm PNG dosyaları başarıyla JPEG formatına dönüştürüldü ve silindi!")
