# Report Lab Day 2

MSSV: `2A202602783`  
Ho ten: `NguyenMinhDuc`

## 1. Tom tat

Bao cao nay duoc tao tu log that trong `runs/` va predictions trong `predictions/`.

Trang thai hien tai: pipeline da duoc kiem tra bang smoke run 1 epoch va inference validation. Chua chay du backbone sweep, ablation, final 3 seed va test vi can thoi gian GPU dai. Neu nop ngay, can giu trung thuc rang day la ban code/pipeline + smoke result, chua phai ket qua full rubric.

## 2. Du lieu va thiet lap

- Fold: 0
- Metric chinh: macro-F1
- Split check: train 10501, val 3501, test 3507; union 17509; overlap 0.

## 3. Top cau hinh theo validation

| exp_id | backbone | seed | val_macro_f1 | val_top1 |
| --- | --- | --- | --- | --- |
| SMOKE | efficientnet_b0 | 0 | 0.5464 | 0.6458 |

## 4. So sanh backbone

Chua co du lieu backbone.

## 5. Ablation training

Chua co du lieu ablation.

## 6. Suy luan va latency

| exp_id | method | checkpoint | K | macro_F1_val | top1_val | ECE_val | latency_p50_ms | latency_p95_ms | latency_p99_ms | images_per_s | temperature | relative_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I00_1view | I00_1view |  | 1 | 0.5467 | 0.6461 | 0.0868 | 17.1902 | 22.4033 | 24.9729 | 58.1725 |  | 1 |
| I01_hflip_prob | I01_hflip_prob |  | 2 | 0.5824 | 0.6772 | 0.0209 |  |  |  |  |  | 2 |
| I03_hflip_logit | I03_hflip_logit |  | 2 | 0.5862 | 0.6824 | 0.0529 |  |  |  |  |  | 2 |
| I07_temperature | I07_temperature |  | 1 | 0.5467 | 0.6461 | 0.0201 | 17.1902 | 22.4033 | 24.9729 | 58.1725 | 1.3673 | 1 |
| I08_amp_latency | I08_amp_latency |  | 1 |  |  |  | 22.7878 | 28.7686 | 29.1910 | 43.8832 |  | 1 |

## 7. Final test

Chua co predictions test. Chi chay test sau khi chot cau hinh tren val.

## 8. Han che

Neu chua chay du >=3 seed hoac chua du cac truc thi nghiem, can ghi ro trong bao cao truoc khi nop.
