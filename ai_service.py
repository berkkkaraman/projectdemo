import os
# 👇 BU İKİ SATIR ÇOK ÖNEMLİ! TENSORFLOW IMPORT EDİLMEDEN ÖNCE YAZILMALI
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# --- AYARLAR ---
MODEL_PATH = 'modelim.h5'
IMG_SIZE = (224, 224) 

# 👇 BURAYI GÜNCELLEDİM: 0. sıraya Kanser, 1. sıraya Sağlıklı yazdım.
CLASS_NAMES = ["Kanser", "Sağlıklı"] 
# ----------------

print(f"TensorFlow Version: {tf.__version__}")
print("⏳ Model yükleniyor (Legacy Mode)...")

try:
    # compile=False diyerek gereksiz parametre hatalarını engelliyoruz
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model Başarıyla Yüklendi!")

except Exception as e:
    print("\n❌ HATA: Model yine yüklenemedi.")
    print(f"Hata Detayı: {e}")
    exit()

def prepare_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Resim işleme hatası: {e}")
        raise e

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Resim yok'}), 400
    
    file = request.files['file']
    
    try:
        processed_image = prepare_image(file.read())
        prediction = model.predict(processed_image)
        result = prediction[0].tolist()
        
        # En yüksek ihtimalin indeksini bul (0 veya 1)
        max_index = int(np.argmax(result))
        
        # 👇 YENİ KISIM: İndeksi yazıya çeviriyoruz
        # Eğer max_index 0 ise "Kanser", 1 ise "Sağlıklı" değerini alır.
        if max_index < len(CLASS_NAMES):
            tahmin_adi = CLASS_NAMES[max_index]
        else:
            tahmin_adi = "Bilinmiyor"
        
        return jsonify({
            'status': 'success',
            'tahmin_index': max_index,
            'tahmin_adi': tahmin_adi, # Web sitesi artık bunu okuyacak
            'oranlar': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)