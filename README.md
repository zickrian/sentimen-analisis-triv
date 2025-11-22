# Analisis Sentimen Ulasan Aplikasi Triv

Proyek ini melakukan analisis sentimen terhadap ulasan pengguna aplikasi **Triv** (aplikasi crypto exchange) dari Google Play Store. Proyek ini mencakup web scraping, preprocessing teks, pelabelan sentimen, visualisasi data, dan machine learning untuk mengklasifikasikan sentimen ulasan.

---

## 📋 Deskripsi Proyek

Proyek ini terdiri dari dua komponen utama:

1. **Web Scraping** (`scraping_data.ipynb`): Mengambil ulasan aplikasi Triv dari Google Play Store
2. **Analisis Sentimen** (`Analisa_Sentimen_Triv.ipynb`): Memproses dan menganalisis sentimen dari ulasan yang telah di-scrape

---

## 📁 Struktur File

```
.
├── scraping_data.ipynb           # Notebook untuk scraping ulasan dari Google Play Store
├── Analisa_Sentimen_Triv.ipynb   # Notebook utama untuk analisis sentimen
├── ulasan_aplikasi.csv           # Dataset ulasan hasil scraping (~1000 ulasan)
├── requirements.txt              # Daftar dependencies Python
└── README.md                     # Dokumentasi proyek (file ini)
```

---

## 🔧 Teknologi dan Library yang Digunakan

### Data Collection
- **google-play-scraper**: Scraping ulasan dari Google Play Store

### Data Processing & Analysis
- **pandas**: Manipulasi dan analisis data
- **numpy**: Komputasi numerik
- **nltk**: Natural Language Toolkit untuk tokenisasi dan stopwords
- **Sastrawi**: Stemming dan stopword removal untuk bahasa Indonesia
- **emoji**: Deteksi dan penanganan emoji

### Visualization
- **matplotlib**: Visualisasi data dasar
- **seaborn**: Visualisasi data statistik
- **wordcloud**: Membuat word cloud dari teks

### Machine Learning
- **scikit-learn**: Machine learning (Random Forest, TF-IDF, evaluasi model)
- **imbalanced-learn**: Menangani data tidak seimbang
- **tensorflow/keras**: Deep learning framework (opsional untuk pengembangan)

### Other Tools
- **Flask**: Framework web (untuk deployment di masa depan)
- **requests**: HTTP requests untuk mengunduh data eksternal

---

## 📊 Workflow Proyek

### 1. Web Scraping (`scraping_data.ipynb`)

**Tujuan**: Mengumpulkan data ulasan aplikasi Triv dari Google Play Store.

**Proses**:
```python
from google_play_scraper import reviews_all, Sort

# Scraping 1000 ulasan aplikasi Triv
scrapreview = reviews_all(
    'id.co.triv',           # ID aplikasi Triv
    lang='id',              # Bahasa Indonesia
    country='id',           # Negara Indonesia
    sort=Sort.MOST_RELEVANT,
    count=1000              # Jumlah maksimum ulasan
)
```

**Output**: File `ulasan_aplikasi.csv` berisi kolom:
- `Review`: Konten ulasan pengguna (kolom utama yang digunakan dalam analisis)

---

### 2. Analisis Sentimen (`Analisa_Sentimen_Triv.ipynb`)

#### A. Loading Dataset
- Membaca file CSV hasil scraping
- Menampilkan informasi dataset
- Mengecek missing values dan duplikat
- Membersihkan data (drop NA dan duplikat)

#### B. Text Preprocessing

Tahapan preprocessing yang dilakukan:

1. **Cleaning Text**
   - Menghapus mentions (@username)
   - Menghapus hashtags (#hashtag)
   - Menghapus RT (retweet)
   - Menghapus URL/link
   - Menghapus angka
   - Menghapus karakter spesial dan tanda baca
   - Menghapus spasi berlebih

2. **Case Folding**
   - Mengubah semua huruf menjadi lowercase

3. **Slang Words Normalization**
   - Mengganti kata-kata slang dengan kata standar
   - Contoh: 'gak' → 'tidak', 'bgt' → 'banget', 'apk' → 'aplikasi', dll.
   - Dictionary mencakup 50+ kata slang umum dalam bahasa Indonesia

4. **Tokenization**
   - Memecah teks menjadi list kata-kata individual

5. **Stopwords Removal**
   - Menghapus kata-kata umum yang tidak bermakna (stopwords)
   - Menggunakan stopwords bahasa Indonesia dan Inggris
   - Custom stopwords: 'iya', 'yaa', 'gak', 'nya', 'sih', 'ku', dll.

6. **Emoji Removal**
   - Mendeteksi dan menghapus baris yang mengandung emoji

#### C. Sentiment Labeling

**Metode**: Lexicon-Based Sentiment Analysis

- Menggunakan lexicon positif dan negatif dari GitHub
- Sumber lexicon: [angelmetanosaa/dataset](https://github.com/angelmetanosaa/dataset)
  - `lexicon_positive.csv`: Kata-kata dengan skor positif
  - `lexicon_negative.csv`: Kata-kata dengan skor negatif

**Algoritma**:
```python
def sentiment_analysis_lexicon_indonesia(text):
    score = 0
    for word in text:
        if word in lexicon_positive:
            score += lexicon_positive[word]
        elif word in lexicon_negative:
            score += lexicon_negative[word]
    
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    
    return score, polarity
```

**Kategori Sentimen**:
- **Positive**: Skor > 0
- **Negative**: Skor < 0
- **Neutral**: Skor = 0

#### D. Exploratory Data Analysis (EDA)

1. **Distribusi Sentimen**
   - Pie chart menampilkan proporsi sentimen positive, negative, dan neutral

2. **Word Cloud**
   - Word cloud dari semua ulasan
   - Word cloud khusus untuk ulasan positif
   - Word cloud khusus untuk ulasan negatif

3. **Visualisasi Data**
   - Class distribution (count plot)
   - Text length distribution (histogram)
   - Most frequent words menggunakan TF-IDF score

#### E. Machine Learning Modeling

**Feature Extraction**:
- **TF-IDF Vectorizer**
  - max_features: 20,000
  - ngram_range: (1, 2) - unigrams dan bigrams
  - max_df: 0.9 - mengabaikan kata yang muncul di >90% dokumen

**Model**: Random Forest Classifier

**Parameter**:
- n_estimators: 100
- random_state: 42
- class_weight: 'balanced' (untuk menangani imbalanced data)
- n_jobs: -1 (parallel processing)

**Train-Test Split**:
- Training: 70%
- Testing: 30%
- Stratified split (mempertahankan proporsi kelas)

**Label Encoding**:
- Neutral → 0
- Negative → 1
- Positive → 2

**Evaluasi Model**:
- Accuracy Score (Training & Testing)
- F1 Score (Macro Average)
- Classification Report (Precision, Recall, F1-Score per kelas)

---

## 🚀 Cara Menggunakan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 3. Scraping Data (Opsional)

Jika ingin scraping ulasan baru:

```bash
jupyter notebook scraping_data.ipynb
```

Jalankan semua cell untuk menghasilkan file `ulasan_aplikasi.csv` baru.

### 4. Analisis Sentimen

```bash
jupyter notebook Analisa_Sentimen_Triv.ipynb
```

Jalankan semua cell untuk:
- Memproses data
- Melabel sentimen
- Visualisasi hasil
- Melatih model machine learning

---

## 📈 Hasil dan Insights

### Distribusi Sentimen
Dari analisis ulasan aplikasi Triv:
- **Positive**: Mayoritas pengguna memberikan ulasan positif tentang kemudahan penggunaan, UI yang bagus, dan transaksi yang cepat
- **Negative**: Beberapa pengguna mengalami masalah dengan withdrawal, verifikasi KYC, dan customer service yang lambat
- **Neutral**: Ulasan dengan skor netral

### Kata-Kata Paling Sering Muncul

**Positive Reviews**:
- aplikasi, mudah, bagus, cepat, crypto, transaksi, triv, cocok, pemula

**Negative Reviews**:
- ribet, lambat, gagal, refund, potongan, admin, bermasalah, kecewa

### Performa Model
- Model Random Forest menunjukkan performa yang baik dalam mengklasifikasikan sentimen
- Class balancing membantu menangani distribusi data yang tidak seimbang

---

## 💡 Fitur Utama

1. ✅ **Automated Web Scraping**: Scraping otomatis dari Google Play Store
2. ✅ **Comprehensive Text Preprocessing**: 6 tahap preprocessing lengkap
3. ✅ **Slang Words Dictionary**: 50+ kata slang bahasa Indonesia
4. ✅ **Lexicon-Based Labeling**: Pelabelan otomatis menggunakan lexicon Indonesia
5. ✅ **Rich Visualizations**: Pie chart, word cloud, distribusi data
6. ✅ **Machine Learning Model**: Random Forest dengan TF-IDF features
7. ✅ **Imbalanced Data Handling**: Class weighting untuk data tidak seimbang

---

## 🔮 Pengembangan Selanjutnya

Beberapa ide untuk pengembangan proyek:

1. **Web Application**:
   - Deploy model menggunakan Flask/Streamlit
   - Real-time sentiment analysis interface

2. **Deep Learning Models**:
   - Implementasi LSTM/GRU untuk sequence modeling
   - Fine-tune pre-trained models (IndoBERT, mBERT, atau XLM-RoBERTa untuk bahasa Indonesia)

3. **Advanced Features**:
   - Aspect-based sentiment analysis
   - Sentiment trend analysis over time
   - Comparison dengan kompetitor

4. **Data Enhancement**:
   - Scraping ulasan dari sumber lain (App Store, Twitter, dll.)
   - Augmentasi data untuk meningkatkan performa model

---

## 📝 Catatan Penting

- Dataset berisi sekitar 1000 ulasan aplikasi Triv
- Lexicon yang digunakan berasal dari sumber eksternal (GitHub)
- Model perlu dilatih ulang jika ada penambahan data baru
- Preprocessing disesuaikan dengan karakteristik bahasa Indonesia informal (slang, singkatan)

---

## 👨‍💻 Kontributor

Proyek ini dibuat untuk analisis sentimen ulasan aplikasi crypto exchange Triv.

---

## 📄 Lisensi

Proyek ini dibuat untuk tujuan pembelajaran dan penelitian.

---

## 🙏 Acknowledgments

- **Sastrawi**: Library stemming bahasa Indonesia
- **NLTK**: Natural Language Toolkit
- **google-play-scraper**: Library untuk scraping Google Play Store
- **angelmetanosaa**: Dataset lexicon sentimen bahasa Indonesia

---

**Happy Analyzing! 📊🚀**
