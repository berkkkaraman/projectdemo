const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const app = express();
const upload = multer({ dest: 'uploads/' }); // Geçici depolama
const PORT = 3000;
const PYTHON_API_URL = 'http://127.0.0.1:5000/predict';

app.post('/analiz-et', upload.single('resim'), async (req, res) => {
    // 1. Dosya kontrolü
    if (!req.file) {
        return res.status(400).json({ error: 'Lütfen bir resim yükleyin.' });
    }

    try {
        // 2. Resmi Python servisine göndermek için hazırla
        const form = new FormData();
        form.append('file', fs.createReadStream(req.file.path));

        // 3. Python servisine istek at
        const response = await axios.post(PYTHON_API_URL, form, {
            headers: {
                ...form.getHeaders()
            }
        });

        // 4. Geçici dosyayı sil (Sunucu şişmesin)
        fs.unlinkSync(req.file.path);

        // 5. Sonucu kullanıcıya dön
        res.json(response.data);

    } catch (error) {
        console.error("Hata:", error.message);
        // Geçici dosyayı silmeyi dene
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        
        res.status(500).json({ 
            error: 'Yapay zeka servisine ulaşılamadı.',
            detay: error.response ? error.response.data : error.message 
        });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Node.js Sunucusu çalışıyor: http://localhost:${PORT}`);
});