"""Praktikum CNN Oceanic Life Dataset - EDA, training, evaluation, visualizations.

Usage:
  python train.py --data dataset --epochs 15 --batch-size 32

Folder dataset may be `dataset/kelas/gambar.jpg` or have split folders
such as `dataset/train/kelas/gambar.jpg`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

SOURCE_URL = "https://www.kaggle.com/datasets/cyanex1702/oceanic-life-dataset"
IMG_TYPES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
GENERIC_DIRS = {'train', 'training', 'val', 'valid', 'validation', 'test', 'testing',
                'images', 'image', 'dataset', 'data', 'oceanic_life', 'oceanic life'}
SEED = 42


CLASSES = [
    'Fish', 'Coral', 'Shark', 'Octopuses', 'Crustaceans',
    'Seaweed', 'Sponges', 'Other marine organisms'
]
ALIASES = {
    'fish':'Fish', 'fishes':'Fish', 'coral':'Coral', 'corals':'Coral',
    'shark':'Shark', 'sharks':'Shark', 'octopus':'Octopuses',
    'octopuses':'Octopuses', 'crustacean':'Crustaceans',
    'crustaceans':'Crustaceans', 'seaweed':'Seaweed',
    'seaweeds':'Seaweed', 'sponges':'Sponges', 'sponge':'Sponges',
    'other marine organisms':'Other marine organisms'
}


def infer_class(filepath: Path, base: Path) -> str:
    """Only accept explicitly named taxonomy folders, not generic/unknown dirs."""
    dirs = filepath.relative_to(base).parts[:-1]
    labels = [ALIASES.get(' '.join(re.findall(r'[a-z0-9]+', x.casefold()))) for x in dirs]
    labels = [label for label in labels if label]
    return labels[-1] if len(set(labels)) == 1 and labels else ''


def scan_dataset(root: Path, out: Path, img_size: int):
    """Collect valid images; cache uniformly sized JPEGs; skip byte-identical copies."""
    paths = sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMG_TYPES),
                   key=lambda p: str(p).lower())
    if not paths:
        parent = root.parent
        proposal = parent/'Usulan_AI'/'Oceanic_Life-Dataset'
        review_file = parent/'review_labels.csv'
        proposed = sum(1 for fp in proposal.rglob('*') if fp.is_file() and fp.suffix.lower() in IMG_TYPES) if proposal.is_dir() else 0
        raise ValueError(
            f'Tidak ada gambar di folder DATASET FINAL: {root}\n'
            f'Gambar pada folder Usulan_AI (belum disetujui): {proposed}\n'
            f'File review label: {review_file} (ada={review_file.is_file()})\n'
            'Jalankan notebook 01 sampai pengelompokan selesai. Jika ada usulan AI, '
            'periksa label satu per satu, isi kelas_final dan status=DISETUJUI_MANUAL '
            'dalam review_labels.csv lalu jalankan ulang sel Langkah 1 notebook 02. '
            'Jika dataset lain sudah berisi folder kelas resmi, arahkan DATA_DIR ke sana. '
            'Jangan menggunakan label prediksi AI yang belum diverifikasi sebagai ground truth.'
        )
    seen = set()
    entries, errors = [], []
    cache_dir = out / 'cache_rgb'
    for i, fp in enumerate(paths, 1):
        label = infer_class(fp, root)
        if not label:
            errors.append({'file': str(fp), 'alasan': 'Folder kelas tidak ditemukan'})
            continue
        try:
            content_hash = hashlib.sha256(fp.read_bytes()).hexdigest()
            if content_hash in seen:
                errors.append({'file': str(fp), 'alasan': 'Duplikat identik; dilewati'})
                continue
            with Image.open(fp) as image:
                image.load()
                image = ImageOps.exif_transpose(image)
                width, height = image.size
                original_mode = image.mode
                gray = image.convert('L')
                mean_light = float(ImageStat.Stat(gray).mean[0])
                flags = []
                if min(width, height) < img_size:
                    flags.append('resolusi_asli_kecil')
                if mean_light < 25:
                    flags.append('sangat_gelap')
                if mean_light > 230:
                    flags.append('sangat_terang')
                cached = cache_dir / label / f'{content_hash[:20]}.jpg'
                cached.parent.mkdir(parents=True, exist_ok=True)
                if not cached.exists():
                    rgb = image.convert('RGB').resize((img_size, img_size), Image.Resampling.BILINEAR)
                    rgb.save(cached, format='JPEG', quality=93)
            seen.add(content_hash)
            entries.append({
                'path_asli': str(fp), 'path_cache': str(cached), 'kelas': label,
                'lebar_asli': width, 'tinggi_asli': height,
                'mode_asli': original_mode, 'ekstensi': fp.suffix.lower(),
                'rerata_kecerahan': round(mean_light, 2),
                'indikasi_kualitas': '; '.join(flags) if flags else 'tidak_terindikasi',
            })
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            errors.append({'file': str(fp), 'alasan': f'Gagal membaca: {str(exc)[:100]}'})
        if i % 500 == 0:
            print(f'  Memproses {i}/{len(paths)} gambar...')
    df = pd.DataFrame(entries)
    if df.empty or df['kelas'].nunique() < 2:
        raise ValueError(
            'Belum ada minimal dua kelas berisi gambar berlabel valid. '
            'Jalankan notebook 01 untuk membentuk folder Oceanic_Life-Dataset/Fish, '
            'Coral, Shark, dst., kemudian tinjau label usulan AI terlebih dahulu. '
            f'Contoh path dalam folder: {[str(x.relative_to(root)) for x in paths[:4]]}'
        )
    pd.DataFrame(errors, columns=['file', 'alasan']).to_csv(out / 'gambar_dilewati.csv', index=False)
    return df, errors


def save_eda(df: pd.DataFrame, out: Path):
    counts = df['kelas'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(max(9, len(counts) * 0.95), 5.2))
    bars = ax.bar(counts.index, counts.values)
    ax.bar_label(bars, padding=3)
    ax.set(title='Distribusi Gambar per Kelas', xlabel='Kelas organisme laut', ylabel='Jumlah gambar')
    ax.tick_params(axis='x', rotation=33)
    fig.tight_layout()
    fig.savefig(out / 'distribusi_kelas.png', dpi=145)
    plt.close(fig)

    names = sorted(df['kelas'].unique(), key=str.casefold)
    fig, axes = plt.subplots(len(names), 2, figsize=(8, min(3.1 * len(names), 58)), squeeze=False)
    for row, name in enumerate(names):
        subset = df[df.kelas == name].sample(min(2, (df.kelas == name).sum()), random_state=SEED)
        for col, (_, example) in enumerate(subset.iterrows()):
            with Image.open(example['path_asli']) as src:
                axes[row, col].imshow(ImageOps.exif_transpose(src).convert('RGB'))
            axes[row, col].set(title=f'{name} — contoh {col + 1}')
            axes[row, col].axis('off')
        if len(subset) < 2:
            axes[row, 1].axis('off')
    fig.tight_layout()
    fig.savefig(out / 'contoh_tiap_kelas.png', dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(df['lebar_asli'], bins=30)
    axes[0].set(title='Distribusi lebar gambar asli', xlabel='Lebar (piksel)', ylabel='Frekuensi')
    axes[1].hist(df['tinggi_asli'], bins=30)
    axes[1].set(title='Distribusi tinggi gambar asli', xlabel='Tinggi (piksel)', ylabel='Frekuensi')
    fig.tight_layout()
    fig.savefig(out / 'ukuran_gambar.png', dpi=145)
    plt.close(fig)

    qual = df['indikasi_kualitas'].value_counts()
    fig, ax = plt.subplots(figsize=(9, max(3.4, 0.45 * len(qual) + 2)))
    ax.barh(qual.index, qual.values)
    ax.set(title='Indikasi kualitas dasar (heuristik)', xlabel='Jumlah gambar')
    fig.tight_layout()
    fig.savefig(out / 'indikasi_kualitas.png', dpi=145)
    plt.close(fig)


def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    counts = df['kelas'].value_counts()
    if counts.min() < 10:
        raise ValueError('Untuk split stratifikasi 80/10/10, setiap kelas sebaiknya berisi >=10 gambar.')
    ids = np.arange(len(df))
    train_i, hold_i = train_test_split(ids, test_size=0.20, random_state=SEED,
                                       stratify=df['kelas'].to_numpy())
    val_i, test_i = train_test_split(hold_i, test_size=0.50, random_state=SEED,
                                     stratify=df.iloc[hold_i]['kelas'].to_numpy())
    df = df.copy()
    df['split'] = ''
    df.loc[train_i, 'split'] = 'train'
    df.loc[val_i, 'split'] = 'validation'
    df.loc[test_i, 'split'] = 'test'
    return df


def make_tf_dataset(frame, mapping, img_size, batch_size, training=False):
    import tensorflow as tf
    paths = frame['path_cache'].to_numpy(dtype=str)
    labels = frame['kelas'].map(mapping).to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=SEED,
                        reshuffle_each_iteration=True)

    def load_jpeg(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, (img_size, img_size))
        image.set_shape((img_size, img_size, 3))
        # Pixels remain 0..255 here. Normalization to 0..1 happens INSIDE the model.
        return image, label
    return ds.map(load_jpeg, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_cnn(num_classes, img_size, lr):
    import tensorflow as tf
    from tensorflow.keras import layers
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name='gambar_rgb')
    x = layers.Rescaling(1.0 / 255.0, name='normalisasi_0_1')(inputs)
    x = layers.RandomFlip('horizontal', name='augment_flip')(x)
    x = layers.RandomRotation(0.08, name='augment_rotasi')(x)
    x = layers.RandomZoom(0.10, name='augment_zoom')(x)
    x = layers.RandomTranslation(0.06, 0.06, name='augment_shift')(x)
    for filters, conv_name in ((32, 'conv_1'), (64, 'conv_2'), (128, 'conv_3')):
        x = layers.Conv2D(filters, 3, padding='same', activation='relu', name=conv_name)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='prediksi')(x)
    model = tf.keras.Model(inputs, outputs, name='oceanic_life_cnn')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def plot_history(history_frame, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = np.arange(1, len(history_frame) + 1)
    for name in ('accuracy', 'val_accuracy'):
        axes[0].plot(epochs, history_frame[name], marker='.', label=name)
    axes[0].set(title='Kurva Akurasi Pelatihan', xlabel='Epoch', ylabel='Accuracy')
    axes[0].legend()
    for name in ('loss', 'val_loss'):
        axes[1].plot(epochs, history_frame[name], marker='.', label=name)
    axes[1].set(title='Kurva Loss Pelatihan', xlabel='Epoch', ylabel='Loss')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / 'akurasi_loss.png', dpi=150)
    plt.close(fig)


def save_matrix(cm, classes, out):
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / 'confusion_matrix.csv', index_label='Label asli')
    fig, ax = plt.subplots(figsize=(max(7, len(classes)), max(6, len(classes) * 0.8)))
    im = ax.imshow(cm, cmap='Blues')
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           xlabel='Prediksi CNN', ylabel='Label asli', title='Confusion Matrix pada Data Test')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    if len(classes) <= 20:
        threshold = cm.max() * 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > threshold else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / 'confusion_matrix.png', dpi=150)
    plt.close(fig)


def gradcam_heatmap(model, image_tensor, target_index=None):
    """Grad-CAM explanation: attention heatmap, NOT calibrated feature importance."""
    import tensorflow as tf
    feature_model = tf.keras.Model(model.inputs, [model.get_layer('conv_3').output, model.output])
    image_tensor = tf.convert_to_tensor(image_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        activations, probs = feature_model(image_tensor, training=False)
        if target_index is None:
            target_index = tf.argmax(probs[0])
        loss = probs[:, target_index]
    grads = tape.gradient(loss, activations)
    channel_weights = tf.reduce_mean(grads, axis=(1, 2))
    heat = tf.reduce_sum(activations[0] * channel_weights[0], axis=-1)
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def overlay_cam(image, heatmap, alpha=0.40):
    rgb = Image.fromarray(np.asarray(image, dtype=np.uint8), 'RGB')
    heat_u8 = np.uint8(255 * heatmap)
    coloring = plt.get_cmap('jet')(heat_u8)[:, :, :3]
    heat_img = Image.fromarray((coloring * 255).astype(np.uint8)).resize(rgb.size, Image.Resampling.BILINEAR)
    return Image.blend(rgb, heat_img, alpha)


def samples_visualization(test_frame, classes, true, preds, model, img_size, out):
    count = min(12, len(test_frame))
    if count == 0:
        return
    sampled = np.random.default_rng(SEED).choice(len(test_frame), size=count, replace=False)
    fig, axes = plt.subplots(math.ceil(count / 4), 4, figsize=(13, 3.3 * math.ceil(count / 4)), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')
    for idx, row_id in enumerate(sampled):
        with Image.open(test_frame.iloc[row_id]['path_cache']) as pic:
            axes.flat[idx].imshow(pic.convert('RGB'))
        axes.flat[idx].set_title(f'Asli: {classes[true[row_id]]}\nPred: {classes[preds[row_id]]}',
                                 fontsize=9, color='green' if true[row_id] == preds[row_id] else 'firebrick')
    fig.tight_layout()
    fig.savefig(out / 'prediksi_sampel_test.png', dpi=145)
    plt.close(fig)

    cam_count = min(4, count)
    fig, axes = plt.subplots(cam_count, 2, figsize=(8.8, 3 * cam_count), squeeze=False)
    for idx, row_id in enumerate(sampled[:cam_count]):
        with Image.open(test_frame.iloc[row_id]['path_cache']) as pic:
            arr = np.array(pic.convert('RGB'))
        heatmap = gradcam_heatmap(model, arr[None, ...].astype('float32'), int(preds[row_id]))
        axes[idx, 0].imshow(arr)
        axes[idx, 0].set_title(f'Gambar asli — {classes[true[row_id]]}')
        axes[idx, 1].imshow(overlay_cam(arr, heatmap))
        axes[idx, 1].set_title(f'Grad-CAM — {classes[preds[row_id]]}')
        for ax in axes[idx]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / 'gradcam_sampel.png', dpi=145)
    plt.close(fig)


def generate_report(summary, metrics, out):
    """Write a report draft only after real metrics have been calculated."""
    counts = summary['kelas_dan_jumlah']
    distribution = '\n'.join(f'| {k} | {v} |' for k, v in counts.items())
    scores = '\n'.join(f'| {k.capitalize()} | {metrics[k]:.4f} |' for k in ('accuracy', 'precision', 'recall', 'f1_score'))
    report = f'''# LAPORAN PRAKTIKUM\n\n## 1. Judul dan Tujuan\n**Judul:** Klasifikasi Jenis Organisme Laut Menggunakan Convolutional Neural Network (CNN) dan Dashboard Streamlit.\n\n**Tujuan:** Mempelajari eksplorasi data citra, preprocessing, pelatihan CNN, evaluasi model, dan pengembangan dashboard klasifikasi gambar.\n\n## 2. Deskripsi Dataset dan Sumber\nSumber: {SOURCE_URL}\nJumlah gambar valid (setelah menghapus duplikat identik): **{summary['jumlah_gambar']}**. Jumlah kelas: **{summary['jumlah_kelas']}**.\n\n| Kelas | Jumlah gambar |\n|---|---:|\n{distribution}\n\n## 3. EDA\nDistribusi kelas dapat dilihat pada `distribusi_kelas.png`. Contoh tiap kelas berada pada `contoh_tiap_kelas.png`; variasi ukuran pada `ukuran_gambar.png`, dan indikasi kualitas sederhana pada `indikasi_kualitas.png`. Evaluasi ketimpangan antarkelas dilakukan dengan rasio kelas terbesar/terkecil sebesar **{summary['rasio_kelas_terbesar_terkecil']:.2f}**. Pemeriksaan kualitas bersifat heuristik, bukan verifikasi manual setiap foto.\n\n## 4. Pra-pemrosesan\nSeluruh citra valid dikonversi menjadi RGB dan di-resize ke **{summary['ukuran_gambar_cnn']} × {summary['ukuran_gambar_cnn']}** piksel. Nilai piksel dinormalisasi ke 0–1 di dalam model melalui lapisan Rescaling. Label kelas dienkode sebagai integer dan model dilatih dengan sparse categorical crossentropy. Data dipisah secara stratified dengan seed {SEED}: **{summary['jumlah_train']} training, {summary['jumlah_validation']} validation, {summary['jumlah_test']} testing** (sekitar 80/10/10). Pembesaran data dilakukan pada training berupa flip, rotation, zoom, dan translation.\n\n## 5. Arsitektur CNN dan Parameter\nModel CNN dari awal: tiga blok Conv2D (32/64/128 filter, kernel 3×3), BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense(128, ReLU), Dropout(0.35), dan Dense({summary['jumlah_kelas']}, Softmax). Adam learning rate **{summary['learning_rate']}**, batch size **{summary['batch_size']}**, maksimal **{summary['maks_epoch']} epoch**, EarlyStopping monitor validation loss.\n\n## 6. Evaluasi Data Test\nHasil di bawah ini berasal dari model yang telah dilatih, dievaluasi pada set testing yang tidak digunakan saat training/validation. Nilai Precision/Recall/F1 menggunakan **weighted average**.\n\n| Metrik | Nilai |\n|---|---:|\n{scores}\n\nPer kelas: `classification_report.csv`.\n\n## 7. Visualisasi dan Analisis\n- **Confusion matrix:** `confusion_matrix.png`; sel diagonal adalah prediksi benar dan sel di luar diagonal adalah kekeliruan klasifikasi.\n- **Grafik akurasi & loss:** `akurasi_loss.png`; bandingkan kurva train dan validation untuk mengidentifikasi kemungkinan overfitting.\n- **Prediksi test:** `prediksi_sampel_test.png`; menampilkan contoh prediksi benar maupun salah jika terpilih.\n- **Grad-CAM:** `gradcam_sampel.png`; memperkirakan area gambar yang paling berkaitan dengan output model. Grad-CAM tidak membuktikan sebab-akibat.\n\n## 8. Dashboard Streamlit\nDashboard memuat ringkasan dataset, EDA, grafik training, confusion matrix, metrik evaluasi dan prediksi gambar unggahan. **Tambahkan tangkapan layar Streamlit setelah menjalankan `streamlit run app.py`.**\n\n## 9. Kesimpulan\nModel CNN mencapai akurasi testing **{metrics['accuracy'] * 100:.2f}%** dengan F1 weighted **{metrics['f1_score']:.4f}**. Periksa `classification_report.csv` untuk menentukan kelas paling sulit, lalu jelaskan kemungkinan penyebab (kemiripan visual, kualitas gambar, atau ketimpangan jumlah data). Hasil tidak dapat digeneralisasi ke seluruh foto organisme laut tanpa validasi lanjutan.\n\n## Daftar Pustaka\n- cyanex1702. *Oceanic Life Dataset*. Kaggle. {SOURCE_URL}\n- TensorFlow. *Keras: The Sequential model, convolutional layers, and model training documentation*. https://www.tensorflow.org/guide/keras\n- Streamlit. *Streamlit documentation*. https://docs.streamlit.io/\n'''
    per_class_path = out / 'classification_report.csv'
    if per_class_path.exists():
        report_df = pd.read_csv(per_class_path)
        report_rows = report_df[report_df['kelas'].isin(summary['kelas'])].copy()
        if not report_rows.empty:
            weak = report_rows.sort_values('f1-score').iloc[0]
            strong = report_rows.sort_values('f1-score', ascending=False).iloc[0]
            report += (f'\n## 10. Analisis Perbandingan Kelas\n'
                       f'Kelas dengan F1-score tertinggi adalah **{strong["kelas"]}** '
                       f'({strong["f1-score"]:.4f}), sedangkan terendah adalah '
                       f'**{weak["kelas"]}** ({weak["f1-score"]:.4f}). '
                       'Faktor potensial mencakup ketimpangan data, kemiripan karakteristik, '
                       'atau kualitas citra; faktor mana yang benar perlu diverifikasi lewat contoh kesalahan.\n')
    cm_path = out / 'confusion_matrix.csv'
    if cm_path.exists():
        cmf = pd.read_csv(cm_path, index_col=0)
        off = cmf.to_numpy().copy()
        np.fill_diagonal(off, 0)
        if off.max() > 0:
            i, j = np.unravel_index(off.argmax(), off.shape)
            report += (f'\n**Kesalahan klasifikasi terbanyak:** kelas asli {cmf.index[i]} '
                       f'diprediksi menjadi {cmf.columns[j]} sebanyak {int(off[i, j])} citra.\n')
    report += ('\n## 11. Dokumentasi Dashboard\n'
               'Tambahkan screenshot halaman *Dataset & EDA*, *Pelatihan*, '
               '*Evaluasi*, dan *Prediksi Gambar Baru* pada bagian ini. '
               'Setiap grafik di dashboard telah diberi uraian interpretasi.\n'
               '\n**Catatan keabsahan:** Jika label diperoleh dari usulan AI, '
               'pastikan semua gambar yang dipakai adalah label yang sudah ditinjau; '
               'nilai pengujian atas label pseudo otomatis tidak boleh dianggap validasi kebenaran biologi.\n')
    (out / 'LAPORAN_HASIL_OTOMATIS.md').write_text(report, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='CNN Oceanic Life - EDA, training, evaluation')
    parser.add_argument('--data', default='hasil_pengelompokan/Oceanic_Life-Dataset', help='Folder data berlabel yang selesai ditinjau dari notebook 01')
    parser.add_argument('--output', default='output', help='Direktori hasil untuk Streamlit')
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    args = parser.parse_args()
    root = Path(args.data).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Folder dataset belum ada: {root}')
    if args.img_size < 48:
        raise ValueError('--img-size minimal 48')
    out.mkdir(parents=True, exist_ok=True)
    print('1/6 Memeriksa gambar dan menyiapkan cache RGB...')
    df, errors = scan_dataset(root, out, args.img_size)
    print('2/6 Membuat EDA...')
    save_eda(df, out)
    df = stratified_split(df)
    df.to_csv(out / 'dataset_metadata.csv', index=False)
    classes = sorted(df['kelas'].unique(), key=str.casefold)
    mapping = {label: index for index, label in enumerate(classes)}
    split_frames = {s: df[df['split'] == s].reset_index(drop=True)
                    for s in ('train', 'validation', 'test')}
    counts = df['kelas'].value_counts().sort_index().to_dict()
    summary = {
        'dataset_url': SOURCE_URL, 'jumlah_gambar': int(len(df)), 'jumlah_kelas': len(classes),
        'kelas': classes, 'kelas_dan_jumlah': {k: int(v) for k, v in counts.items()},
        'jumlah_train': len(split_frames['train']),
        'jumlah_validation': len(split_frames['validation']),
        'jumlah_test': len(split_frames['test']),
        'jumlah_dilewati_atau_duplikat': len(errors),
        'ukuran_gambar_cnn': args.img_size, 'batch_size': args.batch_size,
        'learning_rate': args.learning_rate, 'maks_epoch': args.epochs,
        'normalisasi': 'Rescaling(1/255) di dalam model',
        'rasio_kelas_terbesar_terkecil': float(max(counts.values()) / min(counts.values())),
        'jumlah_mode_warna_asli': {str(k): int(v) for k, v in df['mode_asli'].value_counts().items()},
    }
    (out / 'dataset_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    print('3/6 Memuat TensorFlow dan membangun CNN...')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    train_ds = make_tf_dataset(split_frames['train'], mapping, args.img_size, args.batch_size, True)
    val_ds = make_tf_dataset(split_frames['validation'], mapping, args.img_size, args.batch_size)
    test_ds = make_tf_dataset(split_frames['test'], mapping, args.img_size, args.batch_size)
    model = build_cnn(len(classes), args.img_size, args.learning_rate)
    model.summary()

    print('4/6 Melatih CNN...')
    weights = None
    train_counts = split_frames['train']['kelas'].value_counts()
    if train_counts.max() / train_counts.min() > 1.5:
        total = len(split_frames['train'])
        weights = {mapping[c]: float(total / (len(classes) * train_counts[c])) for c in classes}
        print('Class weights diaktifkan karena ketimpangan distribusi kelas > 1.5x.')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4,
                                          restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out / 'model_oceanic_cnn.keras'),
                                             monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                               patience=2, min_lr=1e-6),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        class_weight=weights, callbacks=callbacks, verbose=1)
    hist = pd.DataFrame(history.history)
    hist.to_csv(out / 'history.csv', index=False)
    plot_history(hist, out)
    # Load the best checkpoint instead of using the final epoch unconditionally.
    model = tf.keras.models.load_model(out / 'model_oceanic_cnn.keras')

    print('5/6 Evaluasi pada testing (belum pernah digunakan untuk update bobot)...')
    probs = model.predict(test_ds, verbose=1)
    y_true = split_frames['test']['kelas'].map(mapping).to_numpy(dtype=int)
    y_pred = np.argmax(probs, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1),
        'macro_precision': float(m_prec), 'macro_recall': float(m_rec), 'macro_f1': float(m_f1),
        'rerata_confidence_prediksi': float(np.max(probs, axis=1).mean()),
    }
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    report = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                    target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / 'classification_report.csv', index_label='kelas')
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    save_matrix(cm, classes, out)
    examples = split_frames['test'][['path_asli', 'path_cache', 'kelas']].copy()
    examples['prediksi'] = [classes[x] for x in y_pred]
    examples['confidence'] = np.max(probs, axis=1)
    examples['benar'] = examples['kelas'] == examples['prediksi']
    examples.to_csv(out / 'prediksi_data_test.csv', index=False)

    print('6/6 Membuat sampel prediksi, Grad-CAM, dan draf laporan...')
    try:
        samples_visualization(split_frames['test'], classes, y_true, y_pred, model, args.img_size, out)
    except Exception as exc:
        print('Visualisasi sampel/Grad-CAM gagal; metrik tetap tersimpan:', exc)
    generate_report(summary, metrics, out)
    print(f'\nSelesai. Hasil disimpan di: {out}')
    print('METRIK TEST:', metrics)
    print('Buka dashboard dengan: streamlit run app.py')


if __name__ == '__main__':
    main()
