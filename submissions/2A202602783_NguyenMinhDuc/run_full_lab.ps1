$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Set-Location "$PSScriptRoot\code"

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

# 0. Quick pipeline check. Remove this from report; it is only a sanity run.
python experiments.py --mode smoke --batch-size 8
python inference_experiments.py --run-root ../runs/SMOKE/seed0 --batch-size 8 --latency-iters 50
python collect_results.py

# 1. Backbone sweep. Expected to take many hours on RTX 3050 4GB.
python experiments.py --mode backbones --epochs 10 --batch-size 16 --num-workers 2
python collect_results.py

# 2. Training recipe ablation on the selected backbone.
# Change --backbone after reading results.xlsx sheet Backbones if another backbone wins.
python experiments.py --mode training --backbone resnet50 --epochs 10 --batch-size 16 --num-workers 2
python collect_results.py

# 3. Inference experiments on the selected/best validation checkpoint.
# Change run-root to the actual best run from results.xlsx.
python inference_experiments.py --run-root ../runs/B01/seed0 --batch-size 16 --latency-iters 100
python collect_results.py

# 4. Final 3-seed test. Run only after all choices are locked from validation results.
# Change --backbone to the locked final backbone and update Config in experiments.py if the final recipe is not baseline.
# python experiments.py --mode final --backbone resnet50 --epochs 10 --batch-size 16 --num-workers 2
# python collect_results.py
# python ..\..\..\eval.py score --pred "..\predictions\F01_seed*_test.csv" --test-csv "..\..\..\data\labels\test_subset0.csv" --labels "..\..\..\data\labels\labels.csv" --tag F01 --out "..\eval_out"
# python ..\..\..\eval.py grade --final "..\predictions\F01_seed*_test.csv" --baseline "..\predictions\T00FINAL_seed*_test.csv" --test-csv "..\..\..\data\labels\test_subset0.csv" --labels "..\..\..\data\labels\labels.csv" --out "..\eval_out"
