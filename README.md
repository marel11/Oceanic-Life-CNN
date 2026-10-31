# Perbaikan error dataset tidak ditemukan — baca dulu

**Sebab:** `02_Praktikum_CNN_Langkah_1-8.ipynb` mencoba membaca folder `hasil_pengelompokan/Oceanic_Life-Dataset`, tetapi folder tersebut masih kosong. Notebook 01 dapat menaruh semua gambar ke `Usulan_AI` bila dataset Kaggle asli tidak memiliki label resmi yang dikenali. Hal ini **bukan** kegagalan CNN.

**Yang diperbaiki dalam paket ini:**
- Notebook 00 untuk memeriksa jumlah gambar di FINAL, Usulan_AI dan folder Downloads.
- Notebook 02 mencari folder kelas yang valid, membaca status `review_labels.csv`, **otomatis menyalin label yang sudah disetujui** ke FINAL, dan memberikan pesan detail jika final masih kosong.
- Fungsi `scan_dataset()` kini memberikan petunjuk status dan path jika folder final kosong.

**Cara praktis meninjau usulan AI:** setelah menjalankan Notebook 01, jalankan di Terminal VS Code:

```bash
python -m streamlit run 03_Tinjau_Label.py
```

Pilih gambar, cocokkan kelasnya, beri tanda bahwa gambar **sudah diperiksa**, kemudian simpan. Ulangi untuk gambar berikutnya. Lanjutkan Notebook 02 yang menyalin hanya label disetujui. Tidak boleh menyetujui massal tanpa review, karena hasil evaluasi akan bias.

**Langkah praktis:** Buka `00_CEK_FOLDER_SEBELUM_TRAINING.ipynb`. Jika semua gambar berada di `Usulan_AI`, buka `hasil_pengelompokan/review_labels.csv`, periksa gambarnya, lalu isikan `kelas_final` dan `status=DISETUJUI_MANUAL` pada gambar yang sudah diverifikasi; simpan CSV. Buka notebook 02 dan jalankan dari atas. Kalau belum ada `review_labels.csv`, jalankan notebook 01 dahulu sampai sel ekspor. **Jangan jalankan kembali sel ekspor notebook 01 setelah mengedit CSV**, karena dapat menimpa perubahan.

Jika kamu punya folder Kaggle asli dengan subfolder kelas yang telah benar, gunakan `DATA_DIR_MANUAL = Path('/path/ke/Oceanic_Life-Dataset')` pada sel Langkah 1 notebook 02.

---

# Praktikum CNN — Oceanic Life Dataset (Langkah 1–8)

**Sumber Kaggle:** https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset  
**Matakuliah:** Analisis Big Data  
**Mahasiswa:** Mochammad Maril Livio (231230065)

## Urutan penggunaan di MacBook/VS Code

1. Ekstrak paket ZIP ini ke folder misalnya `~/Downloads/Oceanic_Life_CNN_Tugas_Lengkap`.
2. VS Code → File → Open Folder → folder tersebut → Terminal → New Terminal.
3. Gunakan **Python 3.11** agar TensorFlow dan dependency lebih kompatibel:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m ipykernel install --user --name oceanic_cnn --display-name "Python (Oceanic CNN)"
   ```
   Bila `python3.11` tidak ditemukan, gunakan interpreter Python 3.11 dari lingkungan Conda/VS Code.
4. Pilih kernel **Python (Oceanic CNN)** di notebook pertama.
5. Jalankan `01_Kelompokkan_Oceanic_Life.ipynb`. Notebook ini mencari folder `Oceanic_Life-Dataset` lokal atau menggunakan `kagglehub`. Jika source tidak berlabel, ia membuat **usulan AI terpisah** (`Usulan_AI`) dan `review_labels.csv`.
6. **WAJIB bila memakai usulan AI:** tinjau tiap file, isi `kelas_final` dan `status=DISETUJUI_MANUAL`, lalu jalankan sel persetujuan pada notebook 01. Hanya gambar resmi/disetujui masuk folder final `hasil_pengelompokan/Oceanic_Life-Dataset/`.
7. Jalankan `02_Praktikum_CNN_Langkah_1-8.ipynb` dari atas ke bawah (sel install cukup sekali). Folder kelas final minimal **2 kelas**, tiap kelas **>=10 gambar** untuk split berstratifikasi 80/10/10. Idealnya gunakan 8 kelas sesuai tugas.
8. Setelah training selesai, jalankan:
   ```bash
   python -m streamlit run app.py
   ```
9. Ambil screenshot semua tab. Draf laporan di `output/LAPORAN_HASIL_OTOMATIS.md`. Lampirkan screenshot dan ulasan gambar di laporan. Word dapat dibuat dari Markdown menggunakan editor atau Pandoc jika dibutuhkan.

## Struktur data yang diharapkan

```text
Oceanic_Life_CNN_Tugas_Lengkap/
├── 01_Kelompokkan_Oceanic_Life.ipynb
├── 02_Praktikum_CNN_Langkah_1-8.ipynb
├── train.py
├── app.py
├── requirements.txt
├── hasil_pengelompokan/
│   ├── Oceanic_Life-Dataset/
│   │   ├── Fish/
│   │   ├── Coral/
│   │   ├── Shark/
│   │   ├── Octopuses/
│   │   ├── Crustaceans/
│   │   ├── Seaweed/
│   │   ├── Sponges/
│   │   └── Other marine organisms/
│   ├── Usulan_AI/
│   └── review_labels.csv
└── output/                     # dibentuk otomatis setelah training
    ├── dataset_metadata.csv
    ├── dataset_summary.json
    ├── history.csv
    ├── metrics.json
    ├── classification_report.csv
    ├── confusion_matrix.csv
    ├── model_oceanic_cnn.keras
    ├── distribusi_kelas.png
    ├── contoh_tiap_kelas.png
    ├── ukuran_gambar.png
    ├── indikasi_kualitas.png
    ├── akurasi_loss.png
    ├── confusion_matrix.png
    ├── prediksi_sampel_test.png
    ├── gradcam_sampel.png
    └── LAPORAN_HASIL_OTOMATIS.md
```

## Catatan validitas

- Situs Kaggle menuliskan sekitar **7.990 gambar**; angka **7.790** yang sebelumnya disebut tidak dipaksakan. Notebook menghitung gambar yang benar-benar tersedia/valid di laptop.
- Dataset yang seluruh gambarnya tidak berlabel **tidak otomatis menjadi ground truth** jika diprediksi CLIP. Review manual harus dilakukan agar metrik tidak mengevaluasi label buatan model yang belum benar.
- Tidak menggunakan data test untuk training; duplikat yang **identik secara byte** dikeluarkan sebelum pemisahan.
- `ModelCheckpoint` menyimpan model Keras yang mendapat validation loss terbaik.
- Precision/Recall/F1 menggunakan weighted average; macro average juga tersedia.
- Grad-CAM adalah peta keterkaitan visual, bukan causal feature importance definitif.
- Model CNN, metrik, laporan final, dan screenshot **belum tersedia dalam ZIP** sebelum program dijalankan dengan data nyata.

## Alternatif terminal (tanpa menjalankan notebook 02)
```bash
python train.py --data hasil_pengelompokan/Oceanic_Life-Dataset --epochs 15 --batch-size 32 --img-size 128
python -m streamlit run app.py
```
