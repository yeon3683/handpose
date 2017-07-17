import numpy
from data import transformations
from data.basetypes import NamedImgSequence
from data.importers import NYUImporter
from util.handdetector import HandDetector
from util.helpers import shuffle_many_inplace


class Dataset(object):
    """
    Base class for managing data. Used to create training batches.
    """

    def __init__(self, imgSeqs=None):
        """
        Constructor
        """
        if imgSeqs is None:
            self._imgSeqs = []
        else:
            self._imgSeqs = imgSeqs
        self._imgStacks = {}
        self._labelStacks = {}

    @property
    def imgSeqs(self):
        return self._imgSeqs

    def imgSeq(self, seqName):
        for seq in self._imgSeqs:
            if seq.name == seqName:
                return seq
        return []

    @imgSeqs.setter
    def imgSeqs(self, value):
        self._imgSeqs = value
        self._imgStacks = {}

    def load(self):
        objNames = self.lmi.getObjectNames()
        imgSeqs = self.lmi.loadSequences(objNames)
        raise NotImplementedError("Not implemented!")

    def imgStackDepthOnly(self, seqName, normZeroOne=False):
        imgSeq = None
        for seq in self._imgSeqs:
            if seq.name == seqName:
                imgSeq = seq
                break
        if imgSeq is None:
            return []

        if seqName not in self._imgStacks:
            # compute the stack from the sequence
            numImgs = len(imgSeq.data)
            data0 = numpy.asarray(imgSeq.data[0].dpt, 'float32')
            label0 = numpy.asarray(imgSeq.data[0].gtorig, 'float32')
            h, w = data0.shape
            j, d = label0.shape
            imgStack = numpy.zeros((numImgs, 1, h, w), dtype='float32')  # num_imgs,stack_size,rows,cols
            labelStack = numpy.zeros((numImgs, j, d), dtype='float32')  # num_imgs,joints,dim
            for i in xrange(numImgs):
                if normZeroOne:
                    imgD = numpy.asarray(imgSeq.data[i].dpt.copy(), 'float32')
                    imgD[imgD == 0] = imgSeq.data[i].com[2] + (imgSeq.config['cube'][2] / 2.)
                    imgD -= (imgSeq.data[i].com[2] - (imgSeq.config['cube'][2] / 2.))
                    imgD /= imgSeq.config['cube'][2]
                else:
                    imgD = numpy.asarray(imgSeq.data[i].dpt.copy(), 'float32')
                    imgD[imgD == 0] = imgSeq.data[i].com[2] + (imgSeq.config['cube'][2] / 2.)
                    imgD -= imgSeq.data[i].com[2]
                    imgD /= (imgSeq.config['cube'][2] / 2.)

                imgStack[i] = imgD
                labelStack[i] = numpy.clip(numpy.asarray(imgSeq.data[i].gt3Dcrop, dtype='float32') / (imgSeq.config['cube'][2] / 2.), -1, 1)

            self._imgStacks[seqName] = imgStack
            self._labelStacks[seqName] = labelStack

        return self._imgStacks[seqName], self._labelStacks[seqName]


class NYUDataset(Dataset):
    def __init__(self, imgSeqs=None, basepath=None):
        """
        constructor
        """
        super(NYUDataset, self).__init__(imgSeqs)
        if basepath is None:
            basepath = '../../data/NYU/'

        self.lmi = NYUImporter(basepath)
