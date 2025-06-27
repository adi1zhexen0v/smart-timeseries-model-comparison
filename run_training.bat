@echo off
call .venv\Scripts\activate
python scripts/train_lstm.py --model_name lstm_default
python scripts/train_lstm.py --model_name lstm_tuned --units 128 --dropout 0.2 --dense_units 64
python scripts/train_tcn.py --model_name tcn_default
python scripts/train_tcn.py --model_name tcn_tuned --nb_filters 64 --kernel_size 4 --dilations 1,2,4,8 --nb_stacks 2 --dropout 0.2 --dense_units 64
pause
