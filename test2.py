import numpy as np

import tensorflow as tf

from gendata import GenData

import os
import argparse

import _pickle as pickle


def RestoreSess(modelPath):
    sess = tf.InteractiveSession()
    # 通过检查点文件锁定最新的模型
    ckpt = tf.train.get_checkpoint_state(modelPath)
    print('载入已有网络模型和参数')
    # 载入图结构，保存在.meta文件中
    modelSaver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    print('加载还原模型：', ckpt.model_checkpoint_path)
    modelSaver.restore(sess, ckpt.model_checkpoint_path)
    graph = tf.get_default_graph()
    INPUT = graph.get_tensor_by_name('input/INPUT:0')
    INRESULT = graph.get_tensor_by_name('input/INRESULT:0')
    KEEPPROB = graph.get_tensor_by_name('input/KEEPPROB:0')
    OUTPUT = graph.get_tensor_by_name('output/OUTPUT:0')
    TRAINSTEP = graph.get_operation_by_name('train/TRAINSTEP')
    COST = graph.get_tensor_by_name('cost/COST:0')
    TESTCOST = graph.get_tensor_by_name('cost/TESTCOST:0')
    return sess, INPUT, INRESULT, KEEPPROB, OUTPUT, TRAINSTEP, COST, TESTCOST


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--withoutdatapath', '-wodp', required=False, help='withoutdatapath', default=None)
    parser.add_argument('--fmodel', '-fm', required=False, help='fmodel', default=None)
    parser.add_argument('--smodel', '-sm', required=False, help='smodel', default=None)

    args = parser.parse_args()

    modelPath1 = args.fmodel
    modelFile1 = 'mine.ckpt'
    # modelFilePath1 = os.path.join(modelPath1, modelFile1)

    modelPath2 = args.smodel
    modelFile2 = 'mine2.ckpt'
    # modelFilePath2 = os.path.join(modelPath2, modelFile2)

    sess1, INPUT1, INRESULT1, KEEPPROB1, OUTPUT1, TRAINSTEP1, COST1, TESTCOST1 = RestoreSess(modelPath1)

    sess2, INPUT2, INRESULT2, KEEPPROB2, OUTPUT2, TRAINSTEP2, COST2, TESTCOST2 = RestoreSess(modelPath2)

    withoutData = None
    withOutDataFile = None

    if args.withoutdatapath is not None:
        withOutDataFile = args.withoutdatapath

    if withOutDataFile is not None and os.path.exists(withOutDataFile):
        print('排除数据文件存在')

        with open(withOutDataFile, 'rb') as wf:
            # withoutData = pickle.load(wf)
            withOutData = np.load(withOutDataFile)
            withOutUserData = withOutData["aroundData"]
            withOutNextData = withOutData["predictData"]

    while True:

        # 生成一组测试数据
        tables = GenData.genPlayingTable()

        for table in tables:
            userTable = table[0]
            backTable = table[1]
            predictTable = table[2]
            aroundTable = table[3]

            encodeAroundTableData, encodePredictData = GenData.encodeData(aroundTable, predictTable)

            if (withOutUserData is None or
                    (
                            withOutUserData is not None and not (
                            (withOutUserData == encodeAroundTableData).all(1).any()
                    )
                    )
            ):
                withOutUserData = np.append(withOutUserData, [encodeAroundTableData], axis=0)
                withOutNextData = np.append(withOutNextData, [encodePredictData], axis=0)

                testUserData = np.array([encodeAroundTableData])
                testNextDataIn = np.array([[-1, -1]])

                __2, ouputValue2 = sess2.run((COST2, OUTPUT2), feed_dict={INPUT2: testUserData,
                                                                          INRESULT2: testNextDataIn,
                                                                          KEEPPROB2: 1.0})

                __1, ouputValue1 = sess1.run((COST1, OUTPUT1), feed_dict={INPUT1: testUserData,
                                                                          INRESULT1: testNextDataIn,
                                                                          KEEPPROB1: 1.0})

                # print("Total cost:" + str(costValue))

                canSolve = sess2.run(tf.argmax(ouputValue2, 1))
                print("预测值：", str(ouputValue2[0]))
                print(canSolve)
                if canSolve[0] == 0:
                    print("可以解决")
                    # GenData.printTable(userTable)
                    # GenData.printTable(backTable)


                    # print("真实值：", str(predictTable))
                    # print("预测值：", str(ouputValue1))
                    #
                    # # 真实值
                    # testNextData1 = np.array(encodePredictData1)
                    # # print("真实索引：", str(sess1.run(tf.argmax(testNextData1, 1))))
                    # # print("预测索引：", str(sess1.run(tf.argmax(ouputValue1, 1))))

                    testNextData = np.array([encodePredictData])
                    correct_prediction1 = sess1.run(tf.equal(tf.argmax(ouputValue1, 1), tf.argmax(testNextData, 1)))


                    if correct_prediction1[0]:
                        print('地雷预测成功')
                    else:
                        print('地雷预测失败')

                        GenData.printTable(aroundTable)
                        GenData.printTable(ouputValue1)

                        input("回车继续\r\n")
                else:
                    GenData.printTable(aroundTable)
                    print('不能解决')

                    input("回车继续\r\n")
                # else:
                #     print('=')


