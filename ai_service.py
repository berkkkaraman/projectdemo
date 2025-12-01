import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- AYARLAR ---
MODEL_CONFIG = {
    'model_kanser': {
        'dosya': 'modelim.h5',
        'etiketler': ["Kanser", "Sağlıklı"],
        'boyut': (224, 224),
        'renk': 'RGB' # 3 Kanal
    },
    'model_akciger': {
        'dosya': 'lungmodel.h5',
        'etiketler': ["kanser","sağlıklı"], 
        'boyut': (224,224),
        'renk': 'RGB' # Eğer model siyah beyaz ise burayı 'L' yapmalısın!
    }
}
# ---------------

YUKLENEN_MODELLER = {}

print("⏳ Modeller yükleniyor...")
for key, config in MODEL_CONFIG.items():
    try:
        if os.path.exists(config['dosya']):
            print(f"   -> {key} yükleniyor...")
            YUKLENEN_MODELLER[key] = load_model(config['dosya'], compile=False)
        else:
            print(f"⚠️ {config['dosya']} bulunamadı!")
    except Exception as e:
        print(f"❌ {key} HATA: {e}")

print("✅ Hazır!")

def prepare_image(img_bytes, hedef_boyut, renk_modu):
    img = Image.open(io.BytesIO(img_bytes))
    
    # Renk ayarı (RGB = Renkli, L = Siyah Beyaz)
    img = img.convert(renk_modu)
    
    # Boyutlandırma
    img = img.resize(hedef_boyut)
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Eğer siyah beyaz ise (512, 512) -> (512, 512, 1) yapmalıyız
    if renk_modu == 'L':
        img_array = np.expand_dims(img_array, axis=-1)
        
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Resim yok'}), 400
    
    secilen_model_key = request.form.get('model_turu')
    
    if secilen_model_key not in YUKLENEN_MODELLER:
        return jsonify({'error': f"Model bulunamadı: {secilen_model_key}"}), 400

    model = YUKLENEN_MODELLER[secilen_model_key]
    ayarlar = MODEL_CONFIG[secilen_model_key]
    
    try:
        file = request.files['file']
        
        # Resmi hazırla
        processed_image = prepare_image(file.read(), ayarlar['boyut'], ayarlar['renk'])
        
        # 🛠️ DEBUG: Terminale bilgi yazdır (Hatanın sebebi burada görünecek)
        print(f"\n🔍 ANALİZ BAŞLADI: {secilen_model_key}")
        print(f"   Giriş Resmi Şekli (Shape): {processed_image.shape}")
        
        # Tahmin yap
        prediction = model.predict(processed_image)
        result = prediction[0]
        
        print(f"   Model Çıktısı (Raw): {result}") # Modelin ne döndürdüğünü görelim

        # SONUÇ YORUMLAMA (Binary vs Multi-class)
        tahmin_adi = ""
        max_index = 0
        
        # Eğer tek bir çıktı varsa (Örn: [0.98]) -> Bu Binary Classification'dır
        if len(result) == 1:
            skor = result[0]
            max_index = 0 if skor < 0.5 else 1 # 0.5 altı birinci sınıf, üstü ikinci sınıf
            # Binary için etiketler listesinde 2 eleman olmalı
            tahmin_adi = ayarlar['etiketler'][max_index] if max_index < len(ayarlar['etiketler']) else "Bilinmiyor"
            final_result = [float(1-skor), float(skor)] # Oranları [Sınıf0, Sınıf1] formatına çevir
        
        # Eğer çoklu çıktı varsa (Örn: [0.1, 0.8, 0.1])
        else:
            max_index = int(np.argmax(result))
            tahmin_adi = ayarlar['etiketler'][max_index] if max_index < len(ayarlar['etiketler']) else "Bilinmiyor"
            final_result = result.tolist()

        print(f"   ✅ Sonuç: {tahmin_adi} (Index: {max_index})\n")

        return jsonify({
            'status': 'success',
            'secilen_model': secilen_model_key,
            'tahmin_adi': tahmin_adi,
            'tahmin_index': max_index,
            'oranlar': final_result
        })

    except Exception as e:
        print(f"❌ KRİTİK HATA: {str(e)}") # Hatayı terminalde kırmızı gibi düşün
        return jsonify({'error': f"Python Hatası: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)