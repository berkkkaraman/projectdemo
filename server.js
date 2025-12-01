const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const app = express();
app.use(express.static('public')); // HTML'i sunmak için
const upload = multer({ dest: 'uploads/' });
const PORT = 3000;
const PYTHON_API_URL = 'http://127.0.0.1:5000/predict';

app.post('/analiz-et', upload.single('resim'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'Lütfen bir resim yükleyin.' });
    }

    // HTML'den gelen model seçimini al
    const modelTuru = req.body.model_secimi; 

    try {
        const form = new FormData();
        form.append('file', fs.createReadStream(req.file.path));
        // Model türünü Python'a ilet
        form.append('model_turu', modelTuru); 

        const response = await axios.post(PYTHON_API_URL, form, {
            headers: { ...form.getHeaders() }
        });

        fs.unlinkSync(req.file.path);
        res.json(response.data);

    } catch (error) {
        console.error("Hata:", error.message);
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        
        // Hata detayını frontend'e gönder
        res.status(500).json({ 
            error: 'İşlem başarısız.', 
            detay: error.response ? error.response.data : error.message 
        });
    }
});

app.listen(PORT, () => {
    console.log(`🚀 Node.js Sunucusu çalışıyor: http://localhost:${PORT}`);
});