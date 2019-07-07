训练第一个网络

  生成数据：python gendata.py -g 100000 -gdp X:\minesweeper\training.npz

  训练：python train.py -m X:\minesweeper\model\ -gdp X:\minesweeper\training.npz
  查看：tensorboard --logdir=X:\minesweeper\logs

  测试：python test.py -m X:\minesweeper\model\ -wodp X:\minesweeper\training.npz

================================================================================

训练第二个网络

  生成数据：python gendatasecond.py -dp X:\minesweeper\training.npz -op X:\minesweeper\training2.npz -m X:\minesweeper\model\

  训练：python trainsecond.py -m X:\minesweeper\model2\ -gdp X:\minesweeper\training2.npz
  查看：tensorboard --logdir=X:\minesweeper\logs2

  测试：python test2.py -fm X:\minesweeper\model\ -sm X:\minesweeper\model2\ -wodp X:\minesweeper\training.npz
