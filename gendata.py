import random
import time
import numpy as np

import tensorflow as tf

import os
import argparse

import _pickle as pickle
import gc
import mmap

import copy


class GenData(object):
    # 生成一个24*24的棋盘，并且设置最大地雷数目为99
    row = 24
    column = 24
    mineCount = 99

    # 获取学习数据的尺寸，也就是喂食数据的尺寸
    # 这里代表的是获取中心块上下左右2个块
    getDataSize = 2
    inputDataSize = [getDataSize*2+1,getDataSize*2+1]

    # m 迷雾,0 空地,! 边界外,? 计算位置
    code = [
        '1', '2', '3', '4', '5', '6', '7', '8', 'x', 'm', '0', '!', '?'
    ]

    # 上方棋盘块中各种状态的长度
    codeLen = len(code)

    def initTable():

        #初始化一个全部为迷雾的棋盘，这是人类看到的棋盘
        userTable = [['m' for i in range(GenData.column)] for i in range(GenData.row)]

        #初始化一个全部为0的棋盘，这是背后全部打开迷雾的棋盘，用于生成喂食数据用
        backTable = [['0' for i in range(GenData.column)] for i in range(GenData.row)]

        # 随机种子，获取随机数，随机得到地雷数量
        # 尽量让学习的棋盘更丰富
        random.seed(time.time())
        currentMineCount = random.randint(5, GenData.mineCount)
        while currentMineCount > 0:

            #随机地雷的位置，行号和列号
            mineRow = random.randint(0, GenData.row - 1)
            mineColumn = random.randint(0, GenData.column - 1)

            # 如果这个位置是空白，那么设置一个地雷
            # 并且减少待布雷的总数
            if backTable[mineRow][mineColumn] == '0':
                backTable[mineRow][mineColumn] = 'x'
                currentMineCount -= 1

        # 在背后全部迷雾打开地图上，标记附近地雷个数信息
        # 用两个for语句扫描行和列，分别放入x和y
        for x in range(0, GenData.row):
            for y in range(0, GenData.column):

                # 如果地图上x行y列不是地雷
                if backTable[x][y] != 'x':
                    # 附近地雷数计数前重置为0
                    nearMineCount = 0

                    # 扫描这个块附近的8个块的地雷个数，累加
                    for (currentScanX, currentScanY) in GenData.nearArea(x, y):
                        if backTable[currentScanX][currentScanY] == 'x':
                            nearMineCount = nearMineCount + 1

                    # 如果附近地雷大于0，那么标记地雷数量
                    if nearMineCount > 0:
                        backTable[x][y] = str(nearMineCount)

        # 返回最终生成结果
        return userTable, backTable

    # 得到附近区域的函数
    # 这里做成函数，是为了在多个地方调用
    def nearArea(x, y):

        # 该数组用于保存周围8个块的状态
        areaList = []

        # 扫描上下左右两格内的所有格子
        # 也就是中心块周围3x3的9个格子
        for scanXOffset in range(-1, 2):
            for scanYOffset in range(-1, 2):

                # 这里减去了一个中心格子，所以最终返回的是8个格子
                if not (scanXOffset == 0 and scanYOffset == 0):

                    # 转化成在棋盘中的实际位置
                    currentScanX = x + scanXOffset
                    currentScanY = y + scanYOffset

                    # 如果没有超出边界，那么就把这个块
                    if (0 <= currentScanX < GenData.row) and (0 <= currentScanY < GenData.column):
                        areaList.append((currentScanX, currentScanY))
        # 返回周围的8个格子
        return areaList

    def getNearAreaData(userTable, x, y):

        getDataSize = GenData.getDataSize

        areaDataList = [['!' for i in range(getDataSize * 2 + 1)] for i in range(getDataSize * 2 + 1)]

        for scanXOffset in range(-getDataSize, getDataSize + 1):
            for scanYOffset in range(-getDataSize, getDataSize + 1):

                scanXIndex = scanXOffset + getDataSize
                scanYIndex = scanYOffset + getDataSize

                # 不要扫描起始点
                if not (scanXOffset == 0 and scanYOffset == 0):

                    currentScanX = x + scanXOffset
                    currentScanY = y + scanYOffset

                    if (0 <= currentScanX < GenData.row) and (0 <= currentScanY < GenData.column):
                        areaDataList[scanXIndex][scanYIndex] = userTable[currentScanX][currentScanY]
                else:
                    areaDataList[scanXIndex][scanYIndex] = '?'

        return areaDataList

    def isClickable(backTable, userTable):

        clickAblePostion = []
        for x in range(0, GenData.row):
            for y in range(0, GenData.column):
                if (userTable[x][y] == 'm' and backTable[x][y] != 'x'):
                    clickAblePostion.append((x, y))

        return clickAblePostion

    # 从一副棋盘生成在玩的过程中将会遇到的各种需要判断的5x5情况
    def genPlayingTable():

        # 最终返回的棋盘数据
        tables = []

        # 用上面的initTable函数初始化棋盘
        userTable, backTable = GenData.initTable()

        # 利用while Ture做一个不断循环点击
        while True:
            # 查看这个棋盘还有没有可以点击的块，也就是是否结束
            # 返回所有可以点击的块的列表
            clickAblePositions = GenData.isClickable(backTable, userTable)
            if (len(clickAblePositions) > 0):

                # 从上面返回的列表中随机点击可以点击一个块
                random.seed(time.time())
                randomClickAblePosition = random.randint(0, len(clickAblePositions) - 1)
                randomClickX = clickAblePositions[randomClickAblePosition][0]
                randomClickY = clickAblePositions[randomClickAblePosition][1]
                clickResult, userTable, backTable = GenData.clickTable(userTable, backTable,
                    randomClickX, randomClickY)

                # 根据点击后的情况，获取所有数字周围的可点击区域和答案
                predictTables = GenData.genPredictResult(backTable, userTable)
                for predictTable in predictTables:
                    # 下面将 玩家看到的迷雾地图；打开所有迷雾的地图；预测结果；25宫格数据
                    # 加入到函数返回的结果的列表中
                    # 这样就对一个棋盘的各种人类遇到的要判断的情况
                    tables.append([userTable, backTable, predictTable[0], predictTable[1]])
            else:
                #如果棋盘能点击的位置都点完了，结束循环
                break

        # 返回结果，其中包含一个列表
        # 每个列表项为：玩家看到的迷雾地图；打开所有迷雾的地图；预测结果；25宫格数据
        return tables

    def isOnlyIsolate(userTable, clickCount):
        openMaskCount = 0
        for x in range(0, GenData.row):
            for y in range(0, GenData.column):
                if userTable[x][y] != 'm':
                    openMaskCount += 1

        if openMaskCount <= clickCount:
            return True
        else:
            return False

    def genPredictResult(backTable, userTable):

        nextTables = []
        predictReulst = []
        allNextTables = set()

        # 解开所有数字旁边的迷雾，然后作为机器学习的数据
        for x in range(0, GenData.row):
            for y in range(0, GenData.column):

                # 如果是数字
                if userTable[x][y] != 'm' and userTable[x][y] != '0':

                    # 扫描数字周围可以点击的区域
                    for (currentScanX, currentScanY) in GenData.nearArea(x, y):

                        # 没有扫描过的点
                        if ((currentScanX, currentScanY) not in allNextTables):

                            # 如果数字周围的是没有打开过的迷雾
                            if (userTable[currentScanX][currentScanY] == 'm'):

                                # 并且迷雾下面是地雷
                                if backTable[currentScanX][currentScanY] == 'x':
                                    # 将训练数据设置为真实地雷，表示计算机能够计算出来
                                    predictReulst = [0, 1]
                                else:

                                    # 将训练数据设置为真实地雷，表示计算机不用计算
                                    nearData = predictReulst = [1., 0.]

                                nearData = GenData.getNearAreaData(userTable, currentScanX, currentScanY)

                                nextTables.append((predictReulst, nearData))
                                allNextTables.add((currentScanX, currentScanY))

        return nextTables

    def clickTable(userTable, backTable, X, Y):

        newUserTable = copy.deepcopy(userTable)

        # 如果没有迷雾, 被探索过
        if newUserTable[X][Y] != 'm':
            return True, newUserTable, backTable
        else:
            # 如果是地雷
            if backTable[X][Y] == 'x':
                return False, newUserTable, backTable
            else:

                # 如果是空白区域
                if backTable[X][Y] == '0':
                    # 打开迷雾，设置为空白区域
                    newUserTable[X][Y] = '0'

                    for (currentScanX, currentScanY) in GenData.nearArea(X, Y):
                        __, newUserTable, __ = GenData.clickTable(newUserTable, backTable, currentScanX, currentScanY);

                else:
                    # 标记数字的情况
                    newUserTable[X][Y] = backTable[X][Y]

                return True, newUserTable, backTable

    def encodeData(aroundTable, predictTable):

        # 将下面的符号转成numpy的数组
        # '1', '2', '3', '4', '5', '6', '7', '8', 'x', 'm', '0', '!', '?'
        codeArray = np.array(GenData.code, dtype=str)

        # 保存转成0和1后的的最终数组
        # 此时的数组还是5x5的
        aroundTableList = list()
        aroundTableWidth = len(aroundTable)
        aroundTableHeight = len(aroundTable[0])
        for x in range(aroundTableWidth):
            for y in range(aroundTableHeight):
                aroundTableList.append((codeArray == aroundTable[x][y]).astype(float))

        # 将5x5的数组通过reshape转成1维数组，便于喂食机器学习
        aroundTableArray = np.array(aroundTableList)\
            .reshape(aroundTableWidth * aroundTableHeight * GenData.codeLen)

        # 将预测结果转成numpy数组，便于机器学习
        predictArray = np.array(predictTable).astype(float)

        # 返回机器可以学习的 5x5局面和结果，其中只有0和1
        return aroundTableArray, predictArray

    def printTable(table):
        for x in range(0, len(table)):
            for y in range(0, len(table[0])):
                print(table[x][y], end=' ')

            print()

        print()

    def convertToTable(tableArray):
        return np.array(tableArray).reshape([GenData.row, GenData.column])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gendata', '-g', required=False, help='gendata', default=0, type=int)
    parser.add_argument('--datapath', '-gdp', required=False, help='datapath', default=None)
    parser.add_argument('--testdatapath', '-gtdp', required=False, help='testdatapath', default=None)
    parser.add_argument('--withoutdatapath', '-wodp', required=False, help='withoutdatapath', default=None)

    args = parser.parse_args()

    maxGenData = args.gendata

    withOutDataFile = args.withoutdatapath

    genDataFile = None
    if (args.datapath is not None):
        genDataFile = args.datapath

    if (args.testdatapath is not None):
        genDataFile = args.testdatapath

    if (args.withoutdatapath is not None):
        withOutDataFile = args.withoutdatapath

    if genDataFile is None:
        print('请指定 -gdp 或 -gtdp 来生成训练数据或者测试数据')
    else:

        if os.path.exists(genDataFile):
            print('数据文件存在，直接读取再在尾部添加')

            trainingData = np.load(genDataFile)

            trainingAroundData = trainingData["aroundData"]
            trainingPredictData = trainingData["predictData"]

        else:
            trainingAroundData = None
            trainingPredictData = None

        withOutAroundData = None
        if withOutDataFile is not None and os.path.exists(withOutDataFile):
            print('排除数据文件存在，载入该数据')

            withOutData = np.load(withOutDataFile)
            withOutAroundData = withOutData["aroundData"]
            withOutPredictData = withOutData["predictData"]

            print('排除数据文件载入数据完毕')

        print('一共需要生成', maxGenData, '条训练数据')

        genDataCount = 0
        while True:

            if genDataCount >= maxGenData:
                break

            # 生成一组训练数据
            tables = GenData.genPlayingTable()

            for table in tables:

                if genDataCount >= maxGenData:
                    break

                # 迷雾地图
                userTable = table[0]
                # 打开全部迷雾地图
                backTable = table[1]
                # 预测结果
                predictTable = table[2]
                # 25宫格数据
                aroundTable = table[3]

                # 旋转三次
                for rotateTime in range(4):

                    if genDataCount >= maxGenData:
                        break

                    if rotateTime > 0:
                        aroundTable = np.rot90(aroundTable)

                    # 预测值,输入值
                    encodeAroundTableData, encodePredictData = GenData.encodeData(aroundTable, predictTable)

                    if (withOutAroundData is None or \
                            (
                                    withOutAroundData is not None and not (
                                    (withOutAroundData == encodeAroundTableData).all(1).any()
                            )
                            )
                    ):

                        if (trainingAroundData is None or \
                                (
                                        trainingAroundData is not None and not (
                                        (trainingAroundData == encodeAroundTableData).all(1).any()
                                )
                                )
                        ):
                            if (trainingAroundData is None):
                                trainingAroundData = np.array([encodeAroundTableData])
                                trainingPredictData = np.array([encodePredictData])
                            else:
                                trainingAroundData = np.append(trainingAroundData, [encodeAroundTableData], axis=0)
                                trainingPredictData = np.append(trainingPredictData, [encodePredictData], axis=0)

                            genDataCount = genDataCount + 1

                            # print('.', end='')
                            # print(genDataCount)
                            # GenData.printTable(aroundTable)
                            print(encodeAroundTableData)

                            if genDataCount % 1000 == 0:
                                print('生成', genDataCount, '条训练数据并保存, 总训练数', len(trainingAroundData), '条')
                                # with open(genDataFile, 'wb') as wf:
                                # pickle.dump(trainingData, wf)

                                np.savez(genDataFile, aroundData=trainingAroundData, predictData=trainingPredictData)
                                print('保存成功')

                            if genDataCount >= maxGenData:
                                break
                        # else:
                        #     print('_')
                        #     GenData.printTable(aroundTable)
                    # else:
                    #     print('=')
                    #     GenData.printTable(aroundTable)

        print(genDataFile, '文件中生成了', maxGenData, '条数据')
        # print(trainingData)
