const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const app = express();
app.use(express.static('public')); // 👈 BU SATIRI EKLE
const upload = multer({ dest: 'uploads/' }); // Geçici klasör
const PORT = 3000;
const PYTHON_API_URL = 'http://127.0.0.1:5000/predict';

app.post('/analiz-et', upload.single('resim'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'Lütfen bir resim yükleyin.' });
    }

    try {
        // Resmi Python servisine gönder
        const form = new FormData();
        form.append('file', fs.createReadStream(req.file.path));

        const response = await axios.post(PYTHON_API_URL, form, {
            headers: { ...form.getHeaders() }
        });

        // Geçici dosyayı sil
        fs.unlinkSync(req.file.path);

        // Sonucu dön
        res.json(response.data);

    } catch (error) {
        console.error("Hata:", error.message);
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        res.status(500).json({ error: 'Python servisine ulaşılamadı.' });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Node.js Sunucusu çalışıyor: http://localhost:${PORT}`);
});