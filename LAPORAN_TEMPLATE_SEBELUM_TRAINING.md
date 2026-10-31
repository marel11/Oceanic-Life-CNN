# LAPORAN PRAKTIKUM — TEMPLATE
**Klasifikasi Jenis Organisme Laut dengan CNN dan Dashboard Streamlit**

**Nama**: Mochammad Maril Livio  
**NIM**: 231230065  
**Mata kuliah**: Analisis Big Data  
**Dosen**: Putri Yuli Utami, S.Kom., M.Kom.

## 1. Judul dan Tujuan
Klasifikasi jenis organisme laut dengan CNN untuk mempelajari pengolahan citra, pembangunan model, evaluasi dan penyajian dashboard.

## 2. Dataset dan sumber
Sumber: https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset

Isi jumlah gambar dan kelas **berdasarkan hasil notebook**; jangan salin angka perkiraan.

## 3. EDA
Lampirkan distribusi kelas, contoh gambar, ukuran, mode RGB/grayscale, kualitas dan class imbalance. Uraikan setiap grafik.

## 4. Pra-pemrosesan
Resize 128×128 (minimal 48×48), konversi RGB, normalisasi 0–1, encoding numerik, split train/validation/test 80/10/10, augmentasi pada data training.

## 5. Arsitektur dan parameter CNN
Conv 32/64/128 + Batch Normalization + Pooling, Dense ReLU, Softmax; Adam, learning rate 0.001, batch 32, maksimal 15 epoch, early stopping.

## 6. Hasil evaluasi
**Diisi otomatis setelah training**: accuracy, precision, recall, F1 weighted/macro; tabel evaluasi tiap kelas.

## 7. Visualisasi dan analisis
Masukkan confusion matrix, accuracy/loss, prediksi beberapa gambar test, Grad-CAM. Jelaskan kelas dengan kesalahan terbanyak dan berikan interpretasi tiap grafik.

## 8. Dashboard Streamlit
Tempel screenshot tab Dataset & EDA, Pelatihan, Evaluasi, Prediksi Gambar Baru. Berikan penjelasan setiap visual dan uji minimal satu gambar baru.

## 9. Kesimpulan
Isi **berdasarkan metrik sesungguhnya** setelah run (jangan mengarang akurasi).

## Daftar Pustaka
- cyanex1702. Oceanic Life Dataset. Kaggle. https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset
- TensorFlow. Keras guide. https://www.tensorflow.org/guide/keras
- Streamlit. Documentation. https://docs.streamlit.io/
