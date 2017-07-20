import numpy
import matplotlib
import matplotlib.pyplot as plt
import os
import cPickle
import sys
import time
from multiprocessing import Process, Queue, Value
from data.importers import NYUImporter
from data.dataset import NYUDataset
from util.handdetector import HandDetector
import cv2

class RealtimeHandposePipeline(object):
    """
    Realtime pipeline for handpose estimation
    """

    # states of pipeline
    STATE_IDLE = 0
    STATE_INIT = 1
    STATE_RUN = 2

    # different hands
    HAND_LEFT = 0
    HAND_RIGHT = 1

    def __init__(self, config, di):
        """
        Initialize data
        :param di: depth importer
        """

        self.importer = di

        # configuration
        self.config = config
        self.initialconfig = config
        # synchronization between threads
        #self.queue = Queue()
        self.stop = Value('b', False)
        # for calculating FPS
        self.lastshow = time.time()
        # hand left/right
        self.hand = self.HAND_LEFT
        # initial state
        self.state = self.STATE_IDLE
        # hand size estimation
        self.handsizes = []
        self.numinitframes = 50
        # hand tracking or detection
        self.tracking = False
        self.lastcom = (0, 0, 0)

    def detect(self, frame):
        """
        Detect the hand
        :param frame: image frame
        :return: cropped image, transformation, center
        """

        hd = HandDetector(frame, self.config['fx'], self.config['fy'], importer=self.importer)

        doHS = (self.state == self.STATE_INIT)
        if self.tracking and not numpy.allclose(self.lastcom, 0):
            loc, handsz = hd.track(self.lastcom, self.config['cube'], doHandSize=doHS)
        else:
            loc, handsz = hd.detect(size=self.config['cube'], doHandSize=doHS)

        self.lastcom = loc
        """
        if self.state == self.STATE_INIT:
            self.handsizes.append(handsz)
            print numpy.median(numpy.asarray(self.handsizes), axis=0)
        else:
            self.handsizes = []

        if self.state == self.STATE_INIT and len(self.handsizes) >= self.numinitframes:
            self.config['cube'] = tuple(numpy.median(numpy.asarray(self.handsizes), axis=0).astype('int'))
            self.state = self.STATE_RUN
            self.handsizes = []

        if numpy.allclose(loc, 0):
            return numpy.zeros((self.poseNet.cfgParams.inputDim[2], self.poseNet.cfgParams.inputDim[3]), dtype='float32'), numpy.eye(3), loc
        else:
            crop, M, com = hd.cropArea3D(loc, size=self.config['cube'], dsize=(self.poseNet.layers[0].cfgParams.inputDim[2], self.poseNet.layers[0].cfgParams.inputDim[3]))
            com3D = self.importer.jointImgTo3D(com)
            crop[crop == 0] = com3D[2] + (self.config['cube'][2] / 2.)
            crop[crop >= com3D[2] + (self.config['cube'][2] / 2.)] = com3D[2] + (self.config['cube'][2] / 2.)
            crop[crop <= com3D[2] - (self.config['cube'][2] / 2.)] = com3D[2] - (self.config['cube'][2] / 2.)
            crop -= com3D[2]
            crop /= (self.config['cube'][2] / 2.)
            return crop, M, com3D
        """
        crop, M, com = hd.cropArea3D(loc, size=self.config['cube'], dsize=(128, 128))
        com3D = self.importer.jointImgTo3D(com)
        crop[crop == 0] = com3D[2] + (self.config['cube'][2] / 2.)
        crop[crop >= com3D[2] + (self.config['cube'][2] / 2.)] = com3D[2] + (self.config['cube'][2] / 2.)
        crop[crop <= com3D[2] - (self.config['cube'][2] / 2.)] = com3D[2] - (self.config['cube'][2] / 2.)
        crop -= com3D[2]
        crop /= (self.config['cube'][2] / 2.)
        return crop, M, com3D

    def processFiles(self, filenames):
        """
        Run detector from files
        :param filenames: filenames to load
        :return: None
        """

        allstart = time.time()
        if not isinstance(filenames, list):
            raise ValueError("Files must be list of filenames.")

        i = 0
        for f in filenames:
            i += 1
            if self.stop.value:
                break
            # Capture frame-by-frame
            start = time.time()
            frame = self.importer.loadDepthMap(f)
            print("{}ms loading".format((time.time() - start)*1000.))

            startd = time.time()
            crop, M, com3D = self.detect(frame.copy())
            print("{}ms detection".format((time.time() - startd)*1000.))

            #startp = time.time()
            #pose = self.estimatePose(crop) * self.config['cube'][2]/2. + com3D
            #print("{}ms pose".format((time.time() - startp)*1000.))

            # Display the resulting frame
            starts = time.time()
            #img = self.show(frame, pose)
            #img = self.addStatusBar(frame)
            cv2.imshow('frame', frame)
            self.lastshow = time.time()
            cv2.imshow('crop', crop)
            cv2.imwrite("./images/{}frame.png".format(i),frame)
            cv2.imwrite("./images/{}crop.png".format(i),crop*255)
            #self.processKey(cv2.waitKey(1) & 0xFF)
            cv2.waitKey(500)
            print("{}ms display".format((time.time() - starts)*1000.))

            print("-> {}ms per frame".format((time.time() - start)*1000.))

        print("DONE in {}s".format((time.time() - allstart)))
        cv2.destroyAllWindows()

if __name__ == '__main__':
    di = NYUImporter('../data/NYU/')
    Seq2 = di.loadSequence('test_1')
    testSeqs = [Seq2]

    testDataSet = NYUDataset(testSeqs)
    test_data, test_gt3D = testDataSet.imgStackDepthOnly('test_1')

    config = {'fx': 588., 'fy': 587., 'cube': (300, 300, 300)}

    rtp = RealtimeHandposePipeline(config, di)

    # use filenames
    filenames = []
    for i in testSeqs[0].data:
        filenames.append(i.fileName)
    # filenames = sorted(glob.glob('./capture2/*.png'))
    rtp.processFiles(filenames)  # Threaded
