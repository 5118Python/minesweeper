生成数据：python gendata.py -g 100000 -gdp X:\minesweeper\training.npz

训练：python train.py -m X:\minesweeper\model\ -gdp X:\minesweeper\training.npz

测试：python test.py -m X:\minesweeper\model\ -wodp X:\minesweeper\training.npz
