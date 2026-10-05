# Lab Day 2 Submission

- MSSV: `2A202602783`
- Ho ten: `NguyenMinhDuc`
- Thu muc nop: `submissions/2A202602783_NguyenMinhDuc`

Code hoan thien nam trong `code/`. Thu muc nay duoc tao tu `starter/` nhung khong sua `starter/` goc, de test khung cua repo van giu nguyen.

Trang thai hien tai: da hoan thien pipeline code va chay smoke/inference validation tren GPU de kiem tra dinh dang. Cac thuc nghiem day du theo rubric can chay bang `run_full_lab.ps1`; khong dien so lieu gia vao `results.xlsx` hay `report.md`.

## Cai dat

```bash
pip install torch torchvision timm pandas numpy scikit-learn matplotlib pillow openpyxl
# tuy chon de dem GMAC
pip install thop
```

Tren Windows nen bat UTF-8 truoc khi chay test/eval co tieng Viet:

```powershell
$env:PYTHONUTF8='1'
```

## Chay nhanh

Dat du lieu theo dung README goc:

- Anh: `data/images/*.jpg`
- Nhan: `data/labels/train_subset0.csv`, `val_subset0.csv`, `test_subset0.csv`, `labels.csv`

Chay mot baseline:

```bash
cd submissions/work/code
python train.py --set exp_id=B01 backbone=resnet50 seed=0 labels_dir=../../../data/labels images_dir=../../../data/images
```

Voi RTX 3050 4GB nen dung batch nho hon mac dinh:

```bash
python train.py --set exp_id=B01 backbone=resnet50 seed=0 batch_size=16 labels_dir=../../../data/labels images_dir=../../../data/images
```

File sinh ra:

- `runs/<exp_id>/seed<k>/config.json`, `history.csv`, `best.pt`, `val_logits.npy`
- `predictions/<exp_id>_seed<k>_val.csv`
- Neu `save_test_predictions=true`: `predictions/<exp_id>_seed<k>_test.csv`
- `curves/<exp_id>_seed<k>.png`

Vi du chay test chi nen dung sau khi da chot cau hinh tren val:

```bash
python train.py --set exp_id=F01 backbone=resnet50 seed=0 save_test_predictions=true labels_dir=../../../data/labels images_dir=../../../data/images
```

## Scripts tu dong

Kiem tra nhanh 1 epoch:

```bash
python experiments.py --mode smoke --batch-size 8
```

Chay backbone sweep toi thieu:

```bash
python experiments.py --mode backbones --epochs 10 --batch-size 16
```

Chay ablation tren backbone da chon:

```bash
python experiments.py --mode training --backbone resnet50 --epochs 10 --batch-size 16
```

Chay final 3 seed va baseline 3 seed, chi lam sau khi da chot cau hinh tren val:

```bash
python experiments.py --mode final --backbone resnet50 --epochs 10 --batch-size 16
```

Chay inference experiments cho mot checkpoint da co:

```bash
python inference_experiments.py --run-root ../runs/B01/seed0 --batch-size 16
```

Tong hop log that vao `results.xlsx` va `report.md`:

```bash
python collect_results.py
```

## Goi y danh sach thi nghiem

Backbone toi thieu:

```text
resnet50
resnext50_32x4d
convnext_tiny
deit_small_patch16_224
swin_tiny_patch4_window7_224
efficientnet_b0
```

Ablation co the chay bang `--set`:

```bash
python train.py --set exp_id=T01 backbone=resnet50 init=frozen seed=0
python train.py --set exp_id=T02 backbone=resnet50 aug=color seed=0
python train.py --set exp_id=T03 backbone=resnet50 loss=ls label_smoothing=0.1 seed=0
python train.py --set exp_id=T04 backbone=resnet50 loss=focal focal_gamma=2.0 seed=0
python train.py --set exp_id=T05 backbone=resnet50 mix=cutmix mix_alpha=1.0 seed=0
python train.py --set exp_id=T06 backbone=resnet50 ema_decay=0.999 seed=0
```

Khong sua `eval.py`; file prediction da duoc ghi bang `eval.save_predictions`.
