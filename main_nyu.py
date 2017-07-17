import numpy
import matplotlib
matplotlib.use('Agg')  # plot to file
import matplotlib.pyplot as plt
import os
import cPickle
import sys
from data.importers import NYUImporter
from data.dataset import NYUDataset
import cv2

if __name__ == '__main__':


    #if not os.path.exists('./eval/'+eval_prefix+'/'):
    #    os.makedirs('./eval/'+eval_prefix+'/')

    rng = numpy.random.RandomState(320)

    print("create data")

    di = NYUImporter('../data/NYU/')
    Seq1 = di.loadSequence('train', shuffle=True, rng=rng)
    trainSeqs = [Seq1]

    Seq2_1 = di.loadSequence('test_1')
    #Seq2_2 = di.loadSequence('test_2')
    #testSeqs = [Seq2_1, Seq2_2]
    testSeqs = [Seq2_1]

    # create training data
    trainDataSet = NYUDataset(trainSeqs)
    train_data, train_gt3D = trainDataSet.imgStackDepthOnly('train', normZeroOne=True)

    mb = (train_data.nbytes) / (1024 * 1024)
    print("data size: {}Mb".format(mb))

    # create validation data
    #valDataSet = NYUDataset(testSeqs)
    #val_data, val_gt3D = valDataSet.imgStackDepthOnly('test_1')

    # create test data
    testDataSet = NYUDataset(testSeqs)
    test_data1, test_gt3D1 = testDataSet.imgStackDepthOnly('test_1', normZeroOne=True)
    #test_data2, test_gt3D2 = testDataSet.imgStackDepthOnly('test_2')

    print train_gt3D.max(), test_gt3D1.max(), train_gt3D.min(), test_gt3D1.min()
    print train_data.max(), test_data1.max(), train_data.min(), test_data1.min()

    imgSizeW = train_data.shape[3]
    imgSizeH = train_data.shape[2]
    nChannels = train_data.shape[1]

    for i in range(1, 10):
        cv2.imshow('image', train_data[i][0])
        cv2.imwrite("{}F.png".format(i),train_data[i][0]*256)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
