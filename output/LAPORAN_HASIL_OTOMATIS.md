# LAPORAN PRAKTIKUM

## 1. Judul dan Tujuan
**Judul:** Klasifikasi Jenis Organisme Laut Menggunakan Convolutional Neural Network (CNN) dan Dashboard Streamlit.

**Tujuan:** Mempelajari eksplorasi data citra, preprocessing, pelatihan CNN, evaluasi model, dan pengembangan dashboard klasifikasi gambar.

## 2. Deskripsi Dataset dan Sumber
Sumber: https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset
Jumlah gambar valid (setelah menghapus duplikat identik): **1022**. Jumlah kelas: **8**.

| Kelas | Jumlah gambar |
|---|---:|
| Coral | 92 |
| Crustaceans | 102 |
| Fish | 331 |
| Octopuses | 47 |
| Other marine organisms | 152 |
| Seaweed | 114 |
| Shark | 122 |
| Sponges | 62 |

## 3. EDA
Distribusi kelas dapat dilihat pada `distribusi_kelas.png`. Contoh tiap kelas berada pada `contoh_tiap_kelas.png`; variasi ukuran pada `ukuran_gambar.png`, dan indikasi kualitas sederhana pada `indikasi_kualitas.png`. Evaluasi ketimpangan antarkelas dilakukan dengan rasio kelas terbesar/terkecil sebesar **7.04**. Pemeriksaan kualitas bersifat heuristik, bukan verifikasi manual setiap foto.

## 4. Pra-pemrosesan
Seluruh citra valid dikonversi menjadi RGB dan di-resize ke **128 × 128** piksel. Nilai piksel dinormalisasi ke 0–1 di dalam model melalui lapisan Rescaling. Label kelas dienkode sebagai integer dan model dilatih dengan sparse categorical crossentropy. Data dipisah secara stratified dengan seed 42: **817 training, 102 validation, 103 testing** (sekitar 80/10/10). Pembesaran data dilakukan pada training berupa flip, rotation, zoom, dan translation.

## 5. Arsitektur CNN dan Parameter
Model CNN dari awal: tiga blok Conv2D (32/64/128 filter, kernel 3×3), BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense(128, ReLU), Dropout(0.35), dan Dense(8, Softmax). Adam learning rate **0.001**, batch size **32**, maksimal **15 epoch**, EarlyStopping monitor validation loss.

## 6. Evaluasi Data Test
Hasil di bawah ini berasal dari model yang telah dilatih, dievaluasi pada set testing yang tidak digunakan saat training/validation. Nilai Precision/Recall/F1 menggunakan **weighted average**.

| Metrik | Nilai |
|---|---:|
| Accuracy | 0.1359 |
| Precision | 0.0939 |
| Recall | 0.1359 |
| F1_score | 0.0460 |

Per kelas: `classification_report.csv`.

## 7. Visualisasi dan Analisis
- **Confusion matrix:** `confusion_matrix.png`; sel diagonal adalah prediksi benar dan sel di luar diagonal adalah kekeliruan klasifikasi.
- **Grafik akurasi & loss:** `akurasi_loss.png`; bandingkan kurva train dan validation untuk mengidentifikasi kemungkinan overfitting.
- **Prediksi test:** `prediksi_sampel_test.png`; menampilkan contoh prediksi benar maupun salah jika terpilih.
- **Grad-CAM:** `gradcam_sampel.png`; memperkirakan area gambar yang paling berkaitan dengan output model. Grad-CAM tidak membuktikan sebab-akibat.

## 8. Dashboard Streamlit
Dashboard memuat ringkasan dataset, EDA, grafik training, confusion matrix, metrik evaluasi dan prediksi gambar unggahan. **Tambahkan tangkapan layar Streamlit setelah menjalankan `streamlit run app.py`.**

## 9. Kesimpulan
Model CNN mencapai akurasi testing **13.59%** dengan F1 weighted **0.0460**. Periksa `classification_report.csv` untuk menentukan kelas paling sulit, lalu jelaskan kemungkinan penyebab (kemiripan visual, kualitas gambar, atau ketimpangan jumlah data). Hasil tidak dapat digeneralisasi ke seluruh foto organisme laut tanpa validasi lanjutan.

## Daftar Pustaka
- cyanex1702. *Oceanic Life Dataset*. Kaggle. https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset
- TensorFlow. *Keras: The Sequential model, convolutional layers, and model training documentation*. https://www.tensorflow.org/guide/keras
- Streamlit. *Streamlit documentation*. https://docs.streamlit.io/

## 10. Analisis Perbandingan Kelas
Kelas dengan F1-score tertinggi adalah **Shark** (0.2281), sedangkan terendah adalah **Coral** (0.0000). Faktor potensial mencakup ketimpangan data, kemiripan karakteristik, atau kualitas citra; faktor mana yang benar perlu diverifikasi lewat contoh kesalahan.

**Kesalahan klasifikasi terbanyak:** kelas asli Fish diprediksi menjadi Shark sebanyak 33 citra.

## 11. Dokumentasi Dashboard
Tambahkan screenshot halaman *Dataset & EDA*, *Pelatihan*, *Evaluasi*, dan *Prediksi Gambar Baru* pada bagian ini. Setiap grafik di dashboard telah diberi uraian interpretasi.

**Catatan keabsahan:** Jika label diperoleh dari usulan AI, pastikan semua gambar yang dipakai adalah label yang sudah ditinjau; nilai pengujian atas label pseudo otomatis tidak boleh dianggap validasi kebenaran biologi.
