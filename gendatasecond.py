import numpy as np
import os

import tensorflow as tf

import sys
import argparse

from gendata import GenData

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', '-dp', required=False, help='datapath', default=None)
    parser.add_argument('--outputpath', '-op', required=False, help='outputpath', default=None)
    parser.add_argument('--fmodel', '-m', required=False, help='fmodel', default=None)

    args = parser.parse_args()

    modelPath = args.fmodel

    sess = tf.InteractiveSession()

    # 通过检查点文件找到最新的模型
    ckpt = tf.train.get_checkpoint_state(modelPath)

    if ckpt is None or ckpt.model_checkpoint_path is None:
        print('没有发现第一个网络的模型, 请确认fmodel指定模型路径')
    else:
        # 载入图结构，保存在.meta文件中
        modelSaver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        print('已载入第一个网络模型和参数')

        modelSaver.restore(sess, ckpt.model_checkpoint_path)

        graph = tf.get_default_graph()

        INPUT = graph.get_tensor_by_name('input/INPUT:0')
        INRESULT = graph.get_tensor_by_name('input/INRESULT:0')
        KEEPPROB = graph.get_tensor_by_name('input/KEEPPROB:0')

        OUTPUT = graph.get_tensor_by_name('output/OUTPUT:0')

        TRAINSTEP = graph.get_operation_by_name('train/TRAINSTEP')

        COST = graph.get_tensor_by_name('cost/COST:0')
        TESTCOST = graph.get_tensor_by_name('cost/TESTCOST:0')

        print('已还原模型：', ckpt.model_checkpoint_path)

        # 载入第一个网络的训练数据，用于生成第二个网络的训练数据
        # 输出为第一个网络是否预测准确
        dataPath = None
        if args.datapath is not None:
            dataPath = args.datapath

        if dataPath is None:
            print('请指定 -dp 载入第一个网络的训练数据')
        else:
            firstNetData = np.load(dataPath)

            firstNetAroundData = firstNetData["aroundData"]
            firstNetPredictData = firstNetData["predictData"]

            print('第一个网络的训练数据加载成功：', dataPath)

            outputFile = None
            if args.outputpath is not None:
                outputFile = args.outputpath

            if outputFile is None:
                print('请指定 -gdp 成第二个网络的训练数据')
            else:

                if os.path.exists(outputFile):
                    print('数据文件存在，直接读取再在尾部添加')

                    trainingData = np.load(outputFile)

                    trainingAroundData = trainingData["aroundData"]
                    trainingPredictData = trainingData["predictData"]

                else:
                    trainingAroundData = None
                    trainingPredictData = None

                print('一共需要生成', len(firstNetAroundData), '条训练数据')

                genDataCount = 0

                for firstDataIndex in range(0, len(firstNetAroundData)):

                    if not ((trainingAroundData == firstNetAroundData[firstDataIndex]).all(1).any()):

                        testAroundData = [firstNetAroundData[firstDataIndex]]
                        testPredictData = [firstNetPredictData[firstDataIndex]]

                        ouputValue = sess.run(OUTPUT, feed_dict={INPUT: testAroundData,
                                                                 INRESULT: testPredictData,
                                                                 KEEPPROB: 1.0})

                        realValue = [firstNetPredictData[firstDataIndex].tolist()]

                        # print("真实值：", str(realValue), "预测值：", str(ouputValue))

                        # print("真实索引：", str(sess.run(tf.argmax(realValue, 1))))
                        # print("预测索引：", str(sess.run(tf.argmax(ouputValue, 1))))

                        correct = sess.run(tf.equal(tf.argmax(ouputValue, 1), tf.argmax(realValue, 1)))
                        correct1 = correct[0]

                        if correct1:
                            # 如果预测准确
                            predict_data = [1., 0.]
                            print('.', end='')
                        else:
                            # 如果预测失败
                            predict_data = [0., 1.]
                            print('X', end='')

                        sys.stdout.flush()

                        # print("训练数据：", predict_data)

                        if trainingAroundData is None:
                            trainingAroundData = np.array([firstNetAroundData[firstDataIndex]])
                            trainingPredictData = np.array([predict_data])
                        else:
                            trainingAroundData = np.append(trainingAroundData, [firstNetAroundData[firstDataIndex]], axis=0)
                            trainingPredictData = np.append(trainingPredictData, [predict_data], axis=0)

                        genDataCount = genDataCount + 1

                        if genDataCount % 100 == 0:
                            print('生成', genDataCount, '条训练数据并保存, 总训练数', len(trainingAroundData), '条')
                            # with open(genDataFile, 'wb') as wf:
                            # pickle.dump(trainingData, wf)

                            np.savez(outputFile, aroundData=trainingAroundData, predictData=trainingPredictData)
                            print('保存成功')

                        # # input("回车继续\r\n")
                    else:
                        if firstDataIndex % 1000 == 0:
                            print('=', firstDataIndex, end='')

                            sys.stdout.flush()

