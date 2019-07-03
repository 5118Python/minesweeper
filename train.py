import argparse
import os

import numpy as np
import tensorflow as tf

from gendata import GenData


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)


def get_weight(in_out_shape):
    # var = tf.Variable(tf.random_normal(shape, stddev=0.35), dtype=tf.float32)
    var = tf.Variable(xavier_init(in_out_shape[0], in_out_shape[1]))

    lamada = 0.004
    tf.add_to_collection('COSTL2', tf.contrib.layers.l2_regularizer(lamada)(var))

    return var


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gendata', '-g', required=False, help='gendata', default=0, type=int)
    parser.add_argument('--datapath', '-gdp', required=False, help='datapath', default=None)
    parser.add_argument('--testdatapath', '-gtdp', required=False, help='testdatapath', default=None)

    parser.add_argument('--rmodel', '-m', required=False, help='rmodel', default=None)

    args = parser.parse_args()

    modelPath = args.rmodel
    modelFile = 'mine.ckpt'
    modelFilePath = os.path.join(modelPath, modelFile)

    # 初始化tensorflow会话
    sess = tf.InteractiveSession()

    # 通过检查点文件找到最新的模型
    ckpt = tf.train.get_checkpoint_state(modelPath)

    if ckpt is None or ckpt.model_checkpoint_path is None:
        print('重新训练网络')

        # 5*5局面的1维数组
        inputWidth = GenData.inputDataSize[0]
        inputHeight = GenData.inputDataSize[1]

        # 神经网络的结构，这里用4层隐含层
        # 层数越多越能拟合复杂的情况，具体根据实践确定
        in_units = inputWidth * inputHeight * GenData.codeLen
        h1_units = inputWidth * inputHeight * 8
        h2_units = inputWidth * inputHeight * 6
        h3_units = inputWidth * inputHeight * 4
        h4_units = inputWidth * inputHeight * 2
        out_units = 2

        with tf.name_scope("input"):
            # 输入x
            x = tf.placeholder(tf.float32, [None, in_units], name='INPUT')

            # 输出y_
            y_ = tf.placeholder(tf.float32, [None, out_units], name='INRESULT')

            # Dropout的比例
            keep_prob = tf.placeholder(tf.float32, name="KEEPPROB")

        with tf.name_scope("hidden1"):
            # 隐含层1的权重和偏置

            # W1 = tf.Variable(xavier_init(in_units, h1_units))
            # W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.35), dtype=tf.float32)
            W1 = get_weight([in_units, h1_units])
            b1 = tf.Variable(tf.zeros([h1_units]), dtype=tf.float32)

            # 隐含层1
            # hidden1 = tf.nn.softplus(tf.matmul(x, W1) + b1)
            # hidden1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
            hidden1 = tf.nn.leaky_relu(tf.matmul(x, W1) + b1)
            # hidden1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
            # hidden1 = tf.contrib.layers.maxout(tf.matmul(x, W1) + b1, h1_units)

            # 实现Dropout，随机将一部分节点置为0
            # keep_prob参数即为保留数据而不置为0的比例
            # hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

        with tf.name_scope("hidden2"):
            # 隐含层2的权重和偏置
            # W2 = tf.Variable(tf.zeros([h1_units, h2_units]), dtype=tf.float32)
            # W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.35), dtype=tf.float32)
            W2 = get_weight([h1_units, h2_units])
            b2 = tf.Variable(tf.zeros([h2_units]), dtype=tf.float32)

            # 隐含层2
            # hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, W2) + b2)
            hidden2 = tf.nn.leaky_relu(tf.matmul(hidden1, W2) + b2)
            # hidden2 = tf.nn.tanh(tf.matmul(hidden1, W2) + b2)
            # hidden2 = tf.contrib.layers.maxout(tf.matmul(hidden1_drop, W2) + b2, h2_units)

            # hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

        with tf.name_scope("hidden3"):
            # 隐含层3的权重和偏置
            # W3 = tf.Variable(tf.zeros([h2_units, h3_units]), dtype=tf.float32)
            # W3 = tf.Variable(tf.truncated_normal([h2_units, h3_units], stddev=0.35), dtype=tf.float32)
            W3 = get_weight([h2_units, h3_units])
            b3 = tf.Variable(tf.zeros([h3_units]), dtype=tf.float32)

            # 隐含层2
            # hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, W3) + b3)
            hidden3 = tf.nn.leaky_relu(tf.matmul(hidden2, W3) + b3)
            # hidden3 = tf.nn.tanh(tf.matmul(hidden2, W3) + b3)
            # hidden3 = tf.contrib.layers.maxout(tf.matmul(hidden2_drop, W3) + b3, h3_units)

            # hidden3_drop = tf.nn.dropout(hidden3, keep_prob)

        with tf.name_scope("hidden4"):
            # 隐含层3的权重和偏置
            # W4 = tf.Variable(tf.zeros([h3_units, h4_units]), dtype=tf.float32)
            # W4 = tf.Variable(tf.truncated_normal([h3_units, h4_units], stddev=0.35), dtype=tf.float32)
            W4 = get_weight([h3_units, h4_units])
            b4 = tf.Variable(tf.zeros([h4_units]), dtype=tf.float32)

            # 隐含层2
            # hidden4 = tf.nn.sigmoid(tf.matmul(hidden3, W4) + b4)
            hidden4 = tf.nn.leaky_relu(tf.matmul(hidden3, W4) + b4)
            # hidden4 = tf.nn.tanh(tf.matmul(hidden3, W4) + b4)
            # hidden4 = tf.contrib.layers.maxout(tf.matmul(hidden3_drop, W4) + b4, h4_units)

            # hidden4_drop = tf.nn.dropout(hidden4, keep_prob)

        with tf.name_scope("output"):
            # 输出层
            # W4 = tf.Variable(tf.zeros([h4_units, out_units]), dtype=tf.float32)
            # W4 = tf.Variable(tf.truncated_normal([h4_units, out_units], stddev=0.35), dtype=tf.float32)
            W_output = get_weight([h4_units, out_units])
            b_output = tf.Variable(tf.zeros([out_units]), dtype=tf.float32)

            # 输出层
            output = tf.nn.sigmoid(tf.add(tf.matmul(hidden4, W_output), b_output), name='OUTPUT')
            # output = tf.nn.tanh(tf.add(tf.matmul(hidden4, W_output), b_output) / 2 + 0.5, name='OUTPUT')
            # output = tf.add(tf.matmul(hidden4, W_output), b_output)

        # 使用平方误差作为cost
        # 计算输出 与 输入 只差，在求差的平方，最后求和，即得到平方误差

        with tf.name_scope("cost"):
            # cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output), reduction_indices=[1]))
            # good -》 cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(output, y_), 2.0))
            # cost = tf.reduce_sum(tf.abs(tf.subtract(output, y_)))

            # delta = tf.constant(0.25)
            # loss_huber_vals = tf.multiply(tf.square(delta), tf.sqrt(1.+tf.square((y_-output)/delta))-1.)
            # cost = tf.reduce_sum(loss_huber_vals);

            # 训练损失
            # cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(output, y_), 2.0))
            cost = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=output))

            tf.add_to_collection('COSTL2', cost)

            loss = tf.add_n(tf.get_collection('COSTL2'), name='COST')

            # 在日志中记录每一步学习的损失
            tf.summary.scalar('train_loss', loss)

            # 测试数据损失

            test_lost = tf.add_n(tf.get_collection('COSTL2'), name='TESTCOST')

            tf.summary.scalar('test_loss', test_lost)

        # 优化器对损失函数进行优化

        with tf.name_scope("train"):
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0003)
            # optimizer =tf.train.AdadeltaOptimizer(learning_rate=0.0003)
            optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
            train_step = optimizer.minimize(loss, name="TRAINSTEP")

        with tf.name_scope('validation'):
            correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
            accuracy_func = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            tf.summary.scalar('train_accuracy', accuracy_func)
            tf.summary.scalar('test_accuracy', accuracy_func)

        MaxToKeep = 10
        modelSaver = tf.train.Saver(max_to_keep=MaxToKeep, keep_checkpoint_every_n_hours=0.5)

        # 初始化全部模型参数
        sess.run(tf.global_variables_initializer())
    else:
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

    # ACCURACY = graph.get_operation_by_name('validation/ACCURACY')

    if args.datapath is None:
        trainingDataFile = 'training.npz'
    else:
        trainingDataFile = args.datapath

    testDataFile = None
    if args.testdatapath is not None:
        testDataFile = args.testdatapath

    if os.path.exists(trainingDataFile):
        # with open(trainingDataFile, 'rb') as wf:
        print('加载训练数据集文件：', trainingDataFile)
        # trainingData = pickle.load(wf)
        trainingData = np.load(trainingDataFile)

        trainingUserData = trainingData["aroundData"]
        trainingNextData = trainingData["predictData"]

        if testDataFile is not None and os.path.exists(testDataFile):
            # with open(testDataFile, 'rb') as wf:
            print('加载测试数据集文件：', testDataFile)
            # testData = pickle.load(wf)
            testData = np.load(testDataFile)

            testUserData = testData["aroundData"]
            testNextData = testData["predictData"]
        else:

            allCount = len(trainingUserData)
            testCount = int(allCount/10)
            testStart = len(trainingUserData) - testCount

            print('从',allCount,'条训练数据后截取',testCount,'条测试数据数据')

            # 截取尾部十分之一的训练数据
            testUserData = trainingUserData[testStart:allCount, :]
            testNextData = trainingNextData[testStart:allCount, :]

            trainingUserData = trainingUserData[0:testStart, :]
            trainingNextData = trainingNextData[0:testStart, :]

            print('截取后训练数据为:',len(trainingUserData))

        train_logs_path = "./logs/train"
        test_logs_path = "./logs/test"

        # 总训练样本数
        n_samples = int(len(trainingUserData))
        n_test_samples = int(len(testUserData))

        # 最大训练轮数
        training_epochs = 9999999

        # 每批个数
        batch_size = n_test_samples

        # 每一轮就显示一次损失cost
        display_step = 1

        # test每一轮显示一次test集上的损失cost
        test_display_step = int(n_samples / batch_size)

        # 创建summary来观察损失值
        merged_summary_op = tf.summary.merge_all()

        summary_writer_train = tf.summary.FileWriter(train_logs_path, graph=tf.get_default_graph())
        summary_writer_test = tf.summary.FileWriter(test_logs_path, graph=tf.get_default_graph())

        # for epoch in range(training_epochs):
        lastMinTestCost = 99999999
        epochCounter = 0
        while True:
            if epochCounter > training_epochs:
                break

            batchStartPosition = 0

            while True:
                epochCounter = epochCounter + 1
                if epochCounter > training_epochs:
                    break

                batchEndPosition = batchStartPosition + batch_size

                trainingUserTableData = trainingUserData[batchStartPosition:batchEndPosition]
                trainingNextTableData = trainingNextData[batchStartPosition:batchEndPosition]

                summaryMerged, costValue, optValue = sess.run((merged_summary_op, COST, TRAINSTEP),
                                                              feed_dict={INPUT: trainingUserTableData,
                                                                         INRESULT: trainingNextTableData,
                                                                         KEEPPROB: 0.7})

                # 显示当前的迭代数和这一轮迭代的平均cost
                if epochCounter % display_step == 0:
                    print("训练轮次:", '%04d' % epochCounter, "训练批次：", '%04d' % batchStartPosition, '-',
                          '%04d' % batchEndPosition, "训练误差=", "{:.9f}".format(costValue))
                    summary_writer_train.add_summary(summaryMerged, epochCounter)

                if epochCounter % test_display_step == 0:
                    testSummaryMerged, testCostValue = sess.run((merged_summary_op, TESTCOST),
                                                                feed_dict={INPUT: testUserData, INRESULT: testNextData,
                                                                           KEEPPROB: 1.0})
                    print("测试集误差:" + str(testCostValue))
                    summary_writer_test.add_summary(testSummaryMerged, epochCounter)

                    # 当最后的测试损失大于现在的测试损失，那么就保存
                    if lastMinTestCost > testCostValue:
                        lastMinTestCost = testCostValue
                        modelSaver.save(sess, modelFilePath, global_step=epochCounter)
                        print('最小误差是：', lastMinTestCost, ', 保存模型到：', modelFilePath)

                batchStartPosition = batchStartPosition + batch_size

                if batchStartPosition >= n_samples:
                    break

        summary_writer_train.close()
    else:
        print('请指定训练数据文件')
