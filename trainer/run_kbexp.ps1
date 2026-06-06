python trainer\train_nnue.py --cache data\nnue_big.bin --out nets\kb1_ctrl.bin --samples trainer\s_kb1ctrl.txt --hl 512 --activation screlu --lam 0.9 --kbuckets 1 > trainer\kb1_ctrl.log 2>&1
python trainer\train_nnue.py --cache data\nnue_big.bin --out nets\kb4_exp.bin --hl 512 --activation screlu --lam 0.9 --kbuckets 4 > trainer\kb4_exp.log 2>&1
