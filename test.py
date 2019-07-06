import numpy as np

import tensorflow as tf

from gendata import GenData

import os
import argparse

import _pickle as pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--withoutdatapath', '-wodp', required=False, help='withoutdatapath', default=None)

    parser.add_argument('--rmodel', '-m', required=False, help='rmodel', default=None)

    args = parser.parse_args()

    modelPath = args.rmodel
    modelFile = 'mine.ckpt'
    modelFilePath = os.path.join(modelPath, modelFile)

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

    withoutData = None
    withOutDataFile = None

    if (args.withoutdatapath is not None):
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

        userTable = tables[0][0]
        backTable = tables[0][1]
        predictTable = tables[0][2]
        aroundTable = tables[0][3]

        encodeAroundTableData, encodePredictData = GenData.encodeData(aroundTable, predictTable)

        if (withOutUserData is None or
                (
                        withOutUserData is not None and not (
                        (withOutUserData == encodeAroundTableData).all(1).any()
                )
                )
        ):
            testUserData = np.array([encodeAroundTableData])
            testNextData = np.array([encodePredictData])

            __, ouputValue = sess.run((COST, OUTPUT), feed_dict={INPUT: testUserData,
                                                                 INRESULT: testNextData,
                                                                 KEEPPROB: 1.0})

            # print("Total cost:" + str(costValue))

            GenData.printTable(userTable)
            GenData.printTable(backTable)
            GenData.printTable(aroundTable)

            print("真实值：", str(predictTable))

            print("预测值：", str(ouputValue))

            print("真实索引：", str(sess.run(tf.argmax(testNextData, 1))))
            print("预测索引：", str(sess.run(tf.argmax(ouputValue, 1))))

            correct_prediction = sess.run(tf.equal(tf.argmax(ouputValue, 1), tf.argmax(testNextData, 1)))
            print("预测结果：", str(correct_prediction))

            input("回车继续\r\n")
