# Document Image Classification (SD234504)

Proyek supervised computer vision untuk mengklasifikasikan halaman dokumen hasil scan ke dalam 16 kategori. Dataset kompetisi berasal dari subset RVL-CDIP. Tugas ini membandingkan ResNet50 dan ConvNeXt-Tiny menggunakan PyTorch.

## Isi repository

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_resnet_document_classification.ipynb
│   └── 02_convnext_document_classification.ipynb
├── scripts/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── engine.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── dataset/                         # data lokal, tidak di-commit
├── checkpoints/                     # bobot lokal, tidak di-commit
├── submissions/                     # CSV submission
└── reports/                          # metrik dan visualisasi
```

Semua fungsi dan class dalam `scripts/*.py` memiliki docstring. Dataset dan checkpoint diabaikan Git karena ukurannya besar; notebook, kode, laporan, dan file submission tersedia di repository.

## Tiga eksperimen model

| Model | Pendekatan | Hasil validation | File submission |
|---|---|---:|---|
| ResNet50 | ImageNet pretrained, fine-tuning CNN | Akurasi 82,17% pada split 15% | `submissions/submission.csv` |
| ConvNeXt-Tiny baseline | ImageNet pretrained, fine-tuning awal | Akurasi 80,79% pada split 25% | `submissions/submission_convnext.csv` |
| ConvNeXt-Tiny tuned | Fine-tuning dari checkpoint baseline, regularisasi dan early stopping | Akurasi terbaik 85,19% pada split 25% | `submissions/submission_convnext_tuned.csv` |

Validation loss ConvNeXt tuned adalah 0,7626 dengan label smoothing 0,05. Nilainya tidak dibandingkan langsung dengan loss baseline karena baseline dihitung tanpa label smoothing.

Tuning berhenti pada epoch 3. Pada checkpoint terbaik (epoch 1), akurasi train 90,65% dan validation 85,19% (gap 5,47 poin). Pada epoch 2–3, validation loss tidak mengalahkan epoch 1 dan gap melebar hingga 9,76 poin; scheduler menurunkan learning rate, lalu early stopping menghentikan training. Checkpoint dipilih berdasarkan validation loss terendah, bukan hanya akurasi tertinggi.

Skor public/private Kaggle belum dicantumkan karena bergantung pada submission akun kompetisi.

## Persiapan data

1. Join kompetisi dan terima aturannya di Kaggle.
2. Unduh data lewat tab **Data → Download All**, atau Kaggle CLI:

   ```bash
   kaggle competitions download -c document-image-classification -p dataset
   ```

3. Ekstrak ZIP ke folder `dataset/`. Notebook dan loader mengharapkan CSV berada langsung di `dataset/`, dengan gambar seperti berikut:

   ```text
   dataset/
   ├── train.csv
   ├── test.csv
   ├── sample_submission.csv
   ├── train/train/<nama_kelas>/*.tif
   └── test/test/*.tif
   ```

Akun Kaggle, Kaggle API token, dan persetujuan aturan kompetisi diperlukan untuk mengunduh data melalui CLI. Loader memakai kolom train `filename`, `class_name`, `label`, serta kolom test `id`, `filename`.

## Instalasi

Gunakan Python 3.10 atau lebih baru. Dari root repository:

```bash
python -m venv .venv
```

Aktifkan environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

Pasang PyTorch dan torchvision build CUDA yang sesuai dengan driver NVIDIA dari panduan resmi PyTorch jika ingin memakai GPU. Setelah itu pasang dependensi proyek:

```bash
pip install -r requirements.txt
```

Notebook/script memilih CUDA bila `torch.cuda.is_available()` bernilai true; jika tidak, proses berjalan di CPU. ConvNeXt pretrained weights mungkin diunduh otomatis pada pemakaian pertama.

## Tutorial menjalankan eksperimen

### 1. ResNet50

1. Buka `notebooks/01_resnet_document_classification.ipynb` di Jupyter atau VS Code.
2. Jalankan sel dari atas ke bawah.
3. Notebook memeriksa data, melatih dan mengevaluasi ResNet50, kemudian menghasilkan `submissions/submission.csv`.
4. Unggah CSV tersebut pada halaman **Submit Predictions** kompetisi.

### 2. ConvNeXt-Tiny baseline

`submissions/submission_convnext.csv` adalah hasil eksperimen baseline yang disimpan di repository. Gunakan file ini untuk membandingkan hasil Kaggle dengan ResNet50 dan ConvNeXt tuned. Checkpoint baseline tidak disertakan karena checkpoint model lokal diabaikan Git.

### 3. ConvNeXt-Tiny tuned

Pipeline modular menggunakan split stratified 75/25, augmentasi dokumen ringan, learning rate terpisah untuk backbone dan classifier, AdamW weight decay, label smoothing, `ReduceLROnPlateau`, dan early stopping.

Jalankan training dan prediksi dari root repository:

```bash
python -m scripts.train
python -m scripts.predict
```

Atau buka `notebooks/02_convnext_document_classification.ipynb` dan jalankan semua sel. Outputnya meliputi:

- `submissions/submission_convnext_tuned.csv`
- `reports/convnext_tuned_history.csv`
- `reports/figures/convnext_tuned_examples_per_class.png`
- `reports/figures/convnext_tuned_validation_loss.png`
- `reports/figures/convnext_tuned_confusion_matrix.png`
- `checkpoints/convnext_tiny_tuned_best.pth` (lokal dan diabaikan Git)

Untuk eksperimen tuning baru dari bobot ImageNet, hapus checkpoint baseline dan tuned beserta `reports/convnext_tuned_history.csv`, lalu jalankan kembali pipeline. Parameter training dapat diubah di `scripts/config.py`.

## Membaca hasil evaluasi

- Kurva loss membandingkan train loss dengan validation loss pada tiap epoch. Train loss turun sementara validation loss naik dapat menandakan overfitting.
- Kolom `generalization_gap` pada history adalah selisih akurasi train dan validation.
- Confusion matrix menampilkan label sebenarnya pada sumbu Y dan prediksi pada sumbu X. Nilai diagonal adalah prediksi benar; nilai di luar diagonal menunjukkan kelas yang tertukar.
- Panel contoh menampilkan gambar train dari masing-masing 16 kelas.

Early stopping membantu membatasi overfitting, tetapi tidak menjamin skor private leaderboard. Tinjau metrik validasi dan Kaggle setelah setiap eksperimen.

## Membuat submission

File submission harus berisi dua kolom `id,label`, satu baris untuk setiap sampel test, dengan label integer 0–15. Ketiga CSV di `submissions/` telah mengikuti urutan ID pada `test.csv`. Pilih file dari model yang ingin dikirim dan unggah melalui halaman kompetisi. Kaggle membatasi submission sesuai aturan kompetisi.

## Push perubahan ke GitHub

Dataset, checkpoint, environment lokal, cache, dan PDF tugas masuk `.gitignore`. Sebelum commit, periksa daftar file:

```bash
git status --short
git check-ignore dataset/train.csv checkpoints/convnext_tiny_tuned_best.pth
```

File yang diabaikan tidak ikut di-commit. Jangan menambahkan dataset atau checkpoint secara paksa.
