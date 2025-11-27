const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const app = express();
const upload = multer(); 

const PORT = 3000;

const MODEL_YOLU = 'file://./model_klasoru/model.json'; 

let model;

// Modeli Yükle
async function loadModel() {
    try {
        model = await tf.loadLayersModel(MODEL_YOLU);
        console.log("✅ Yapay Zeka Modeli Hazır!");
    } catch (error) {
        console.error("❌ Model yüklenemedi. Klasör yolunu kontrol et:", error);
    }
}
loadModel();

// Resmi İşle
function processImage(buffer) {
    // Resmi tensora çevir
    let tensor = tf.node.decodeImage(buffer, 3);
    // Boyutlandır (Senin modelin 224x224 ise burayı 224 yap)
    tensor = tf.image.resizeBilinear(tensor, [224, 224]); 
    tensor = tensor.div(255.0); // Normalize et
    tensor = tensor.expandDims(0); // [1, 224, 224, 3] formatına getir
    return tensor;
}

// Sunucu İsteği
app.post('/analiz-et', upload.single('resim'), async (req, res) => {
    if (!model) return res.status(500).json({ error: 'Model yükleniyor...' });
    
    try {
        const tensor = processImage(req.file.buffer);
        const prediction = model.predict(tensor);
        const result = prediction.dataSync();
        
        // En yüksek sonucu bul
        const maxIndex = result.indexOf(Math.max(...result));
        
        res.json({
            sonuc_index: maxIndex,
            tum_oranlar: Array.from(result)
        });
    } catch (e) {
        console.error(e);
        res.status(500).send("Hata oluştu: " + e.message);
    }
});

app.listen(PORT, () => console.log(`🚀 Sunucu çalışıyor: http://localhost:${PORT}`));