"""Langkah 7 - Dashboard Streamlit untuk proyek Oceanic Life CNN.

Letakkan app.py di folder yang SAMA dengan notebook dan folder output/.
Jalankan: python -m streamlit run app.py

Membaca hasil Langkah 2-6 tanpa membutuhkan dataset_summary.json dari Langkah 8.
Model berasal dari model_oceanic_cnn.keras (normalisasi 1/255 di dalam model).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
MODEL_PATH = OUT / "model_oceanic_cnn.keras"
KAGGLE_URL = "https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset"

st.set_page_config(page_title="Oceanic Life | CNN", page_icon="🌊", layout="wide")
st.title("🌊 Klasifikasi Organisme Laut Menggunakan CNN")
st.caption("Praktikum Analisis Big Data | Langkah 7 — Dashboard Streamlit")
st.markdown(f"**Sumber dataset:** [Oceanic Life Dataset (Kaggle)]({KAGGLE_URL})")


def load_csv(name, **kwargs):
    path = OUT / name
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"Tidak dapat membaca {name}: {exc}")
        return None


def load_json(name):
    path = OUT / name
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.warning(f"Tidak dapat membaca {name}: {exc}")
        return {}


def show_picture(name, caption):
    path = OUT / name
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
        return True
    st.info(f"Gambar {name} belum tersedia. Periksa sel Langkah 2 atau 6.")
    return False


@st.cache_resource(show_spinner="Memuat model CNN...")
def load_model(model_path):
    import tensorflow as tf
    return tf.keras.models.load_model(str(model_path), compile=False)


metadata = load_csv("dataset_metadata.csv")
history = load_csv("history.csv")
metrics = load_json("metrics.json")
summary = load_json("dataset_summary.json")
reports = load_csv("classification_report.csv")
predictions = load_csv("prediksi_data_test.csv")

if metadata is None and not summary:
    st.error("Belum menemukan output/dataset_metadata.csv atau output/dataset_summary.json.")
    st.code(f"Folder yang dicari: {OUT}\nLetakkan app.py satu folder dengan notebook yang berhasil dijalankan.")
    st.stop()

# Metadata dari Langkah 3 dapat menggantikan ringkasan Langkah 8.
counts = (
    metadata["kelas"].value_counts().sort_index()
    if metadata is not None and "kelas" in metadata.columns
    else pd.Series(summary.get("kelas_dan_jumlah", {}), dtype="int64")
)
classes = (
    sorted(metadata["kelas"].dropna().astype(str).unique(), key=str.casefold)
    if metadata is not None and "kelas" in metadata.columns
    else list(summary.get("kelas", []))
)
if not classes:
    st.error("Daftar kelas tidak ditemukan dalam metadata dataset.")
    st.stop()

if metadata is not None and "split" in metadata.columns:
    split_counts = metadata["split"].value_counts().to_dict()
else:
    split_counts = {
        "train": summary.get("jumlah_train", 0),
        "validation": summary.get("jumlah_validation", 0),
        "test": summary.get("jumlah_test", 0),
    }

is_pseudo = summary.get("sumber_label") == "pseudo"
if not summary.get("sumber_label") and metadata is not None and "path_asli" in metadata.columns:
    is_pseudo = metadata["path_asli"].astype(str).str.contains("Usulan_AI", regex=False).any()

if is_pseudo:
    st.warning(
        "⚠️ Mode percobaan — label gambar berasal dari **usulan AI (CLIP)** yang belum diperiksa. "
        "Akurasi/F1 menunjukkan kecocokan CNN dengan label AI, **bukan akurasi terhadap label biologis yang tervalidasi**."
    )
else:
    st.info("Periksa apakah seluruh label benar-benar telah diverifikasi sebelum memakai hasil sebagai evaluasi final.")

st.sidebar.header("Informasi Praktikum")
st.sidebar.metric("Gambar valid", f"{int(counts.sum()):,}")
st.sidebar.metric("Jumlah kelas", len(classes))
st.sidebar.caption("Data train / validation / test")
for split in ["train", "validation", "test"]:
    st.sidebar.write(f"{split.title()}: **{split_counts.get(split, 0)}**")
st.sidebar.caption(f"Folder hasil: {OUT}")

tab_eda, tab_train, tab_eval, tab_upload = st.tabs([
    "📊 Dataset & EDA", "📈 Pelatihan CNN", "✅ Evaluasi & Visualisasi", "🖼️ Prediksi Upload"
])

with tab_eda:
    st.subheader("Ringkasan Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total gambar valid", f"{int(counts.sum()):,}")
    c2.metric("Jumlah kelas", len(classes))
    c3.metric("Training", int(split_counts.get("train", 0)))
    c4.metric("Testing", int(split_counts.get("test", 0)))
    st.caption(f"Validation: {int(split_counts.get('validation', 0))} gambar. Target pembagian 80% / 10% / 10%.")
    st.dataframe(counts.rename_axis("Kelas").reset_index(name="Jumlah gambar"),
                 hide_index=True, use_container_width=True)

    if not show_picture("distribusi_kelas.png", "Grafik distribusi gambar per kelas"):
        st.bar_chart(counts.rename("Jumlah gambar"))
    if len(counts) and counts.min() > 0:
        ratio = float(counts.max() / counts.min())
        st.write(f"**Analisis distribusi:** Kelas terbanyak memiliki **{ratio:.2f}×** jumlah gambar kelas tersedikit. "
                 + ("Ketidakseimbangan ini perlu diperhatikan ketika menginterpretasikan metrik." if ratio > 1.5
                    else "Distribusi relatif seimbang menurut perbandingan sederhana ini."))

    show_picture("contoh_tiap_kelas.png", "Contoh gambar dari setiap kelas")
    if metadata is not None and {"kelas", "path_asli"}.issubset(metadata.columns):
        st.subheader("Analisis gambar contoh per kelas")
        examples = metadata.groupby("kelas", group_keys=False).sample(n=1, random_state=42)
        for _, row in examples.iterrows():
            left, right = st.columns([1, 2])
            image_path = Path(str(row["path_asli"]))
            with left:
                if image_path.is_file():
                    st.image(str(image_path), caption=str(row["kelas"]), use_container_width=True)
                else:
                    cache_path = Path(str(row.get("path_cache", "")))
                    if cache_path.is_file():
                        st.image(str(cache_path), caption=str(row["kelas"]), use_container_width=True)
                    else:
                        st.caption("File gambar asli tidak ditemukan pada lokasi tersimpan.")
            with right:
                st.markdown(f"**Kelas acuan: {row['kelas']}**")
                st.write(
                    f"Ukuran asli: {row.get('lebar_asli', '-')} × {row.get('tinggi_asli', '-')} piksel. "
                    f"Mode warna: {row.get('mode_asli', '-')}. "
                    f"Indikasi kualitas: {row.get('indikasi_kualitas', '-')}.")
                st.caption("Analisis: Amati bentuk, tekstur, warna, sudut pandang dan latar bawah laut. "
                           "Indikasi kualitas dihitung secara sederhana, bukan pemeriksaan visual ahli.")
    a, b = st.columns(2)
    with a:
        if show_picture("ukuran_gambar.png", "Sebaran ukuran gambar asli"):
            st.caption("Analisis: Ukuran gambar asli berbeda-beda; resize menyamakan ukuran masukan CNN, namun dapat mengurangi detail.")
    with b:
        if show_picture("indikasi_kualitas.png", "Indikasi kualitas gambar"):
            st.caption("Analisis: Gambar terlalu gelap, terlalu terang, atau beresolusi kecil mungkin lebih sulit diklasifikasikan.")

with tab_train:
    st.subheader("Arsitektur dan Hasil Pelatihan")
    st.markdown("**Arsitektur:** Rescaling(1/255) → augmentasi → 3 blok Conv2D (32, 64, 128) "
                "+ BatchNorm + MaxPooling → GlobalAveragePooling → Dense(128) → Dropout → Softmax.")
    st.caption("Optimizer Adam • Learning rate 0.001 • Batch size 32 • Maksimal 15 epoch")
    show_picture("akurasi_loss.png", "Kurva akurasi dan loss training/validation")
    if history is not None and not history.empty:
        st.dataframe(history.round(4), hide_index=True, use_container_width=True)
        if {"accuracy", "val_accuracy", "val_loss"}.issubset(history.columns):
            best_epoch = int(history["val_loss"].idxmin()) + 1
            gap = 100 * (history["accuracy"].iloc[-1] - history["val_accuracy"].iloc[-1])
            st.markdown(f"**Analisis grafik:** Validation loss terbaik tercapai pada epoch **{best_epoch}**. "
                        f"Selisih akurasi train dan validation di akhir pelatihan adalah **{gap:.2f} poin persentase**. "
                        "Jika train terus membaik sedangkan validation memburuk, ada indikasi overfitting. "
                        "Interpretasikan bersama tren seluruh kurva, bukan satu angka saja.")
    else:
        st.info("Riwayat training belum ditemukan: output/history.csv.")

with tab_eval:
    st.subheader("Evaluasi CNN pada Data Testing")
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        for col, key, label in [
            (c1, "accuracy", "Accuracy"), (c2, "precision", "Precision weighted"),
            (c3, "recall", "Recall weighted"), (c4, "f1_score", "F1-score weighted")
        ]:
            value = metrics.get(key)
            col.metric(label, f"{float(value) * 100:.2f}%" if value is not None else "—")
        st.caption("Weighted average memperhitungkan banyaknya gambar per kelas. "
                   "Nilai merupakan perbandingan terhadap label dataset yang digunakan.")
        if metrics.get("macro_f1") is not None:
            st.write(f"Macro F1-score: **{float(metrics['macro_f1']) * 100:.2f}%**")
    else:
        st.warning("File output/metrics.json belum ditemukan. Jalankan Langkah 5 terlebih dahulu.")

    show_picture("confusion_matrix.png", "Confusion matrix: baris adalah label acuan, kolom adalah prediksi")
    cm_frame = load_csv("confusion_matrix.csv", index_col=0)
    if cm_frame is not None and not cm_frame.empty:
        cm = cm_frame.to_numpy()
        wrong = cm.copy()
        np.fill_diagonal(wrong, 0)
        if wrong.max() > 0:
            i, j = np.unravel_index(wrong.argmax(), wrong.shape)
            st.markdown(f"**Analisis confusion matrix:** Kesalahan paling sering dari **{cm_frame.index[i]}** "
                        f"menjadi **{cm_frame.columns[j]}** sebanyak **{wrong[i,j]} gambar**. "
                        "Periksa gambar kedua kelas tersebut untuk memahami kesalahan.")
        else:
            st.write("Pada data test ini, seluruh prediksi sesuai label acuan. Uji dengan data baru tetap diperlukan.")

    if reports is not None:
        st.subheader("Precision, Recall dan F1-Score tiap kelas")
        st.dataframe(reports.round(4), hide_index=True, use_container_width=True)
        if {"kelas", "f1-score"}.issubset(reports.columns):
            eligible = reports[reports["kelas"].isin(classes)]
            if not eligible.empty:
                low = eligible.loc[eligible["f1-score"].idxmin()]
                st.write(f"**Analisis per kelas:** Nilai F1 terendah pada **{low['kelas']}** "
                         f"({low['f1-score']:.3f}). Periksa kemiripan visual, jumlah gambar, dan kemungkinan label keliru.")

    show_picture("prediksi_sampel_test.png", "Prediksi beberapa gambar data testing")
    if predictions is not None and not predictions.empty:
        st.subheader("Analisis tiap gambar uji terpilih")
        shown = predictions.sample(min(6, len(predictions)), random_state=42)
        for idx, (_, row) in enumerate(shown.iterrows(), 1):
            left, right = st.columns([1, 2])
            with left:
                path = Path(str(row.get("path_asli", "")))
                if not path.is_file():
                    path = Path(str(row.get("path_cache", "")))
                if path.is_file():
                    st.image(str(path), use_container_width=True)
                else:
                    st.caption("Gambar tidak ditemukan.")
            with right:
                is_match = str(row["kelas"]) == str(row["prediksi"])
                st.write(f"**Gambar {idx}** — Label acuan: **{row['kelas']}**, "
                         f"prediksi: **{row['prediksi']}**")
                st.write(f"Skor Softmax: {100 * float(row['confidence']):.2f}%")
                st.write("**Analisis:** Prediksi sesuai label acuan; bukan jaminan label biologis benar."
                         if is_match else
                         "**Analisis:** Prediksi berbeda dari label acuan. Bandingkan bentuk organisme, "
                         "latar, kualitas foto, dan kemungkinan kesalahan label.")
    if show_picture("gradcam_sampel.png", "Grad-CAM pada contoh gambar testing"):
        st.caption("Analisis Grad-CAM: area lebih terang/hangat menunjukkan bagian citra dengan kontribusi lebih kuat "
                   "menurut metode visualisasi ini. Bukan penjelasan sebab-akibat yang pasti.")

with tab_upload:
    st.subheader("Prediksi Langsung dari Gambar Baru")
    st.write("Upload gambar organisme laut (JPG, PNG, atau WEBP) untuk memprediksi salah satu dari delapan kelas.")
    if not MODEL_PATH.is_file():
        st.error(f"Model belum ditemukan di {MODEL_PATH}. Selesaikan Langkah 4 dan simpan model .keras.")
    else:
        uploaded = st.file_uploader("Pilih gambar organisme laut", type=["jpg", "jpeg", "png", "webp"])
        if uploaded is not None:
            try:
                image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
                model = load_model(str(MODEL_PATH))
                height, width = model.input_shape[1:3]
                if height is None or width is None:
                    height = width = 128
                resized = image.resize((width, height), Image.Resampling.BILINEAR)
                x = np.asarray(resized, dtype=np.float32)[None, ...]
                # Model Langkah 4 sudah memiliki layer Rescaling(1/255).
                # JANGAN normalisasi dengan /255 dua kali.
                probabilities = np.asarray(model.predict(x, verbose=0)[0], dtype=float)
                if len(probabilities) != len(classes):
                    st.error("Jumlah kelas model berbeda dari metadata. Pastikan output dan model berasal dari training yang sama.")
                else:
                    index = int(np.argmax(probabilities))
                    left, right = st.columns(2)
                    with left:
                        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
                    with right:
                        st.metric("Hasil prediksi CNN", classes[index])
                        st.metric("Skor Softmax", f"{100 * probabilities[index]:.2f}%")
                        scores = pd.DataFrame({"Kelas": classes, "Skor": probabilities})
                        st.bar_chart(scores.set_index("Kelas")["Skor"])
                    st.markdown(f"**Analisis gambar:** Model mengklasifikasikan foto sebagai **{classes[index]}** "
                                f"(skor Softmax {probabilities[index] * 100:.2f}%). "
                                "Skor Softmax bukan jaminan probabilitas benar. "
                                "Bandingkan foto dengan kategori lain bila objek kurang jelas atau latar mendominasi.")
                    try:
                        from train import gradcam_heatmap, overlay_cam
                        heatmap = gradcam_heatmap(model, x, index)
                        cam = overlay_cam(np.asarray(resized), heatmap)
                        st.image(cam, caption="Grad-CAM untuk gambar unggahan", width=390)
                        st.caption("Analisis: Perhatikan apakah area hangat mengenai organisme atau latar air/karang. "
                                   "Visualisasi tidak menjamin model memfokuskan perhatian secara biologis benar.")
                    except Exception as exc:
                        st.info(f"Prediksi berhasil; Grad-CAM tidak dapat ditampilkan ({exc}).")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                st.error(f"File tidak bisa diproses sebagai gambar: {exc}")
            except Exception as exc:
                st.error(f"Gagal memprediksi. Pastikan model kompatibel dan TensorFlow terpasang: {exc}")

st.divider()
st.caption("Dashboard menggunakan keluaran eksperimen yang tersimpan. "
           "Jika label masih usulan AI, hasil belum merupakan validasi terhadap ground truth.")
