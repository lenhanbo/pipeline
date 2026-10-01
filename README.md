# Handwriting GAN pipeline

Baseline sinh chữ viết tay tiếng Anh theo phong cách của một writer. Mô hình sẽ
được huấn luyện trên IAM ở mức word; paragraph được tạo bằng cách sinh từng word
rồi ghép chúng bằng một layout engine.

## 0. Cài môi trường

Project dùng Conda env `pipeline`, Python 3.11 và PyTorch có CUDA 12.8. Trên
Windows/PowerShell:

```powershell
conda activate pipeline
python -m pip install -r requirements.txt
```

Weights & Biases là tùy chọn; TensorBoard đã có sẵn trong bộ cài chính:

```powershell
python -m pip install -r requirements-optional.txt
```

Kiểm tra PyTorch đã nhận GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 1. Chuẩn bị IAM

Đăng ký và tải dữ liệu từ trang IAM chính thức:

https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database

Chỉ cần tải hai archive:

- `data/words.tgz`
- `data/ascii.tgz`

Giải nén để có cấu trúc:

```text
data/raw/iam/
|-- ascii/
|   |-- forms.txt
|   `-- words.txt
`-- words/
    `-- a01/
        `-- a01-000u/
            `-- a01-000u-00-00.png
```

Trên PowerShell, sau khi đặt hai archive vào `data/raw/iam/`:

```powershell
tar -xzf data/raw/iam/words.tgz -C data/raw/iam
tar -xzf data/raw/iam/ascii.tgz -C data/raw/iam
```

Tạo metadata và writer-disjoint splits:

```powershell
python scripts/prepare_iam.py --iam-root data/raw/iam --output data/processed/iam
```

Output gồm `train.jsonl`, `val.jsonl`, `test.jsonl` và `summary.json`. Mỗi record
có `image_path`, `text`, `writer_id`, `form_id`, và `word_id`.

## Pipeline dự kiến

```text
text -> character embedding ----+
                                +-> Transformer -> CNN decoder -> word image
style images -> CNN encoder -----+                         ^
                                                           |
                                                GAN discriminator
```

Paragraph renderer sẽ dùng chung style embedding cho tất cả word, tự wrap dòng
và ghép các ảnh word lên một canvas trắng.
