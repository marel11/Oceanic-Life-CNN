"""Review CLIP-proposed labels with a manual verification checkpoint.
Run: streamlit run 03_Tinjau_Label.py
This tool NEVER auto-approves images or assigns ground-truth labels by itself.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
CSV_PATH = PROJECT_DIR / 'hasil_pengelompokan' / 'review_labels.csv'
CLASSES = [
    'Fish', 'Coral', 'Shark', 'Octopuses', 'Crustaceans',
    'Seaweed', 'Sponges', 'Other marine organisms'
]

st.set_page_config(page_title='Tinjau Label Oceanic Life', layout='wide')
st.title('Pemeriksaan Label Oceanic Life')
st.write(
    'Periksa setiap gambar dan tentukan kelas yang sesuai. Label dari CLIP hanyalah usulan. '
    'Hanya gambar yang sudah Anda periksa dan setujui yang akan digunakan untuk melatih CNN.'
)

if not CSV_PATH.is_file():
    st.error('review_labels.csv belum ada. Jalankan Notebook 01 sampai tahap ekspor.')
    st.code(str(CSV_PATH))
    st.stop()

try:
    data = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, encoding='utf-8-sig')
except Exception as exc:
    st.error(f'Tidak bisa membaca CSV: {exc}')
    st.stop()

required = {'status', 'kelas_final', 'kelas_usulan', 'file_hasil', 'path_sumber'}
if not required.issubset(data.columns):
    st.error(f'Kolom CSV tidak lengkap: {sorted(required-data.columns)}')
    st.stop()

status = data['status'].str.strip().str.upper()
pending = status.eq('PERLU_REVIEW_AI')
approved = status.isin(['DISETUJUI_MANUAL', 'DISETUJUI_DARI_LABEL_ASLI'])
c1,c2,c3=st.columns(3)
c1.metric('Total file dalam daftar', len(data))
c2.metric('Usulan yang belum diperiksa', int(pending.sum()))
c3.metric('Label yang disetujui', int(approved.sum()))

mode = st.radio('Tampilkan', ['Belum diperiksa', 'Sudah disetujui', 'Semua gambar'], horizontal=True)
if mode == 'Belum diperiksa':
    pool = list(data.index[pending])
elif mode == 'Sudah disetujui':
    pool = list(data.index[approved])
else:
    pool = list(data.index)

page_size = 9
max_page = max(1, (len(pool)+page_size-1)//page_size)
page = st.number_input('Halaman', min_value=1, max_value=max_page, value=1, step=1)
selection = pool[(page-1)*page_size:page*page_size]

if not selection:
    st.info('Tidak ada gambar pada daftar yang dipilih.')
    st.stop()

st.caption(f'Menampilkan {len(selection)} gambar pada halaman {page} dari {max_page}')
changes = {}
cols = st.columns(3)
for count, idx in enumerate(selection):
    row = data.loc[idx]
    with cols[count % 3]:
        st.markdown('---')
        photo = Path(row['file_hasil']).expanduser()
        if not photo.is_file():
            photo = Path(row['path_sumber']).expanduser()
        if photo.is_file():
            st.image(str(photo), caption=photo.name, use_container_width=True)
        else:
            st.warning('File gambar tidak ditemukan pada path asal maupun folder usulan.')
        proposal = row['kelas_usulan'] if row['kelas_usulan'] in CLASSES else ''
        st.caption(f'Label usulan AI: {proposal or "Tidak tersedia"}')
        existing = row['kelas_final'] if row['kelas_final'] in CLASSES else proposal
        chosen = st.selectbox('Label setelah diperiksa', options=CLASSES,
                              index=CLASSES.index(existing) if existing in CLASSES else 0,
                              key=f'klass_{idx}')
        old_approved = row['status'].strip().upper() in ('DISETUJUI_MANUAL', 'DISETUJUI_DARI_LABEL_ASLI')
        checked = st.checkbox('Saya sudah memeriksa gambar ini dan menyetujui labelnya',
                              value=old_approved, key=f'ok_{idx}')
        if checked and not old_approved:
            changes[idx] = chosen
        elif checked and old_approved and chosen != row['kelas_final']:
            changes[idx] = chosen

st.warning('Klik Simpan hanya untuk gambar yang benar-benar telah Anda periksa. Jangan menyetujui seluruh usulan otomatis.')
if st.button(f'Simpan label yang telah diperiksa ({len(changes)} gambar)', type='primary', disabled=not changes):
    latest = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if len(latest) != len(data):
        st.error('CSV berubah sejak halaman dibuka. Muat ulang halaman dan periksa kembali.')
        st.stop()
    for row_idx, correct_label in changes.items():
        latest.loc[row_idx, 'kelas_final'] = correct_label
        latest.loc[row_idx, 'status'] = 'DISETUJUI_MANUAL'
    temp = CSV_PATH.with_name('review_labels_tmp.csv')
    latest.to_csv(temp, index=False, encoding='utf-8-sig')
    os.replace(temp, CSV_PATH)
    st.success(f'{len(changes)} label disetujui dan disimpan. Buka notebook 02 untuk menyalin gambar yang disetujui ke folder final.')
    st.rerun()

st.divider()
st.markdown('**Setelah review:** Buka `02_Praktikum_CNN_Langkah_1-8.ipynb` dan jalankan dari atas. Notebook akan menyalin gambar yang telah disetujui ke folder per kelas secara otomatis.')
