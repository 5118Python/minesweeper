生成数据：python gendata.py -g 100000 -gdp F:\minesweeper\training.npz

训练：python train.py -m F:\minesweeper\model\ -gdp F:\minesweeper\training.npz

测试：python test.py -m F:\minesweeper\model\ -wodp F:\minesweeper\training.npz
