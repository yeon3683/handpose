import fnmatch
import scipy.io
import numpy as np
from PIL import Image
import glob
import os
import progressbar as pb
import struct
from data.basetypes import ImgFrame, NamedImgSequence
from util.handdetector import HandDetector
from data.transformations import transformPoint2D
import cPickle
import matplotlib
import matplotlib.pyplot as plt



class DepthImporter(object):
    """
    provide baisc functionality to load depth data
    """

    def __init__(self, fx, fy, ux, uy):
        """
        Initialize object
        :param fx: focal length in x direction
        :param fy: focal length in y direction
        :param ux: principal point in x direction
        :param uy: principal point in y direction
        """

        self.fx = fx
        self.fy = fy
        self.ux = ux
        self.uy = uy

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0]-self.ux)*sample[2]/self.fx
        ret[1] = (sample[1]-self.uy)*sample[2]/self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = sample[1]/sample[2]*self.fy+self.uy
        ret[2] = sample[2]
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x3 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = self.fy
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        print("img min {}, max {}".format(frame.dpt.min(),frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:, 0], frame.gtcrop[:, 1])

        ax.plot(frame.gtcrop[0:4, 0], frame.gtcrop[0:4, 1], c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[4:7, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[4:7, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[7:10, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[7:10, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[10:13, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[10:13, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[0, 0], frame.gtcrop[13:16, 0])),
                np.hstack((frame.gtcrop[0, 1], frame.gtcrop[13:16, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < numcols and row >= 0 and row < numrows:
                z = frame.dpt[row, col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)

        ax.format_coord = format_coord
        plt.show()

class NYUImporter(DepthImporter):
    """
    provide functionality to load data from the NYU hand dataset
    """

    def __init__(self, basepath, useCache=True, cacheDir='./cache/'):
        """
        Constructor
        :param basepath: base path of the dataset
        :return:
        """

        super(NYUImporter, self).__init__(588.03, 587.07, 320., 240.)

        self.basepath = basepath
        self.useCache = useCache
        self.cacheDir = cacheDir
        self.numJoints = 36
        self.scales = {'train': 1., 'test_1': 1., 'test_2': 0.83, 'test': 1., 'train_synth': 1.,
                       'test_synth_1': 1., 'test_synth_2': 0.83, 'test_synth': 1.}
        self.restrictedJointsEval = [1, 4, 5, 7, 10, 11, 13, 16, 17, 19, 22, 23, 25, 26, 28, 33]

    def loadDepthMap(self,filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        img = Image.open(filename)
        # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        assert len(img.getbands()) == 3
        r, g, b = img.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        dpt = np.bitwise_or(np.left_shift(g, 8), b)
        imgdata = np.asarray(dpt, np.float32)

        return imgdata

    def loadSequence(self,seqName,allJoints=False,Nmax=float('inf'),shuffle=False,rng=None,docom=False):
        """
        Load an image sequence from the dataset
        :param seqName: sequence name, e.g. train
        :param subSeq: list of subsequence names, e.g. 0, 45, 122-5
        :param Nmax: maximum number of samples to load
        :return: returns named image sequence
        """

        config = {'cube':(300,300,300)}
        config['cube'] = [s*self.scales[seqName] for s in config['cube']]

        pickleCache = '{}/{}_{}_{}_{}_cache.pkl'.format(self.cacheDir, self.__class__.__name__, seqName, allJoints, docom)
        if self.useCache:
            if os.path.isfile(pickleCache):
                print("Loading cache data from {}".format(pickleCache))
                f = open(pickleCache, 'rb')
                (seqName,data,config) = cPickle.load(f)
                f.close()

                # shuffle data
                if shuffle and rng is not None:
                    print("Shuffling")
                    rng.shuffle(data)
                if not(np.isinf(Nmax)):
                    return NamedImgSequence(seqName,data[0:Nmax],config)
                else:
                    return NamedImgSequence(seqName,data,config)

        # Load the dataset
        objdir = '{}/{}/'.format(self.basepath,seqName)
        trainlabels = '{}/{}/joint_data.mat'.format(self.basepath, seqName)

        mat = scipy.io.loadmat(trainlabels)
        names = mat['joint_names'][0]
        joints3D = mat['joint_xyz'][0]
        joints2D = mat['joint_uvd'][0]
        if allJoints:
            eval_idxs = np.arange(36)
        else:
            eval_idxs = self.restrictedJointsEval

        self.numJoints = len(eval_idxs)

        txt = 'Loading {}'.format(seqName)
        pbar = pb.ProgressBar(maxval=joints3D.shape[0],widgets=[txt, pb.Percentage(), pb.Bar()])
        pbar.start()

        data = []
        i=0
        for line in range(joints3D.shape[0]):
            dptFileName = '{0:s}/synthdepth_1_{1:07d}.png'.format(objdir, line+1)

            if not os.path.isfile(dptFileName):
                print("File {} does not exist!".format(dptFileName))
                i += 1
                continue
            dpt = self.loadDepthMap(dptFileName)

            # joints in image coordinates
            gtorig = np.zeros((self.numJoints, 3), np.float32)
            jt = 0
            for ii in range(joints2D.shape[1]):
                if ii not in eval_idxs:
                    continue
                gtorig[jt,0] = joints2D[line,ii,0]
                gtorig[jt,1] = joints2D[line,ii,1]
                gtorig[jt,2] = joints2D[line,ii,2]
                jt += 1

            # normalized joints in 3D coordinates
            gt3Dorig = np.zeros((self.numJoints,3),np.float32)
            jt = 0
            for jj in range(joints3D.shape[1]):
                if jj not in eval_idxs:
                    continue
                gt3Dorig[jt,0] = joints3D[line,jj,0]
                gt3Dorig[jt,1] = joints3D[line,jj,1]
                gt3Dorig[jt,2] = joints3D[line,jj,2]
                jt += 1
            #print gt3D

            # Detect hand
            hd = HandDetector(dpt, self.fx, self.fy, importer=self)
            if not hd.checkImage(1):
                print("Skipping image {}, no content".format(dptFileName))
                i += 1
                continue
            try:
                if allJoints:
                    dpt, M, com = hd.cropArea3D(gtorig[34], size=config['cube'],docom=False)
                else:
                    dpt, M, com = hd.cropArea3D(gtorig[13], size=config['cube'],docom=False)
            except UserWarning:
                print("Skipping image {}, no hand detected".format(dptFileName))
                continue

            com3D = self.jointImgTo3D(com)
            gt3Dcrop = gt3Dorig - com3D  # normalize to com
            gtcrop = np.zeros((gtorig.shape[0], 3), np.float32)
            for joint in range(gtorig.shape[0]):
                t=transformPoint2D(gtorig[joint], M)
                gtcrop[joint, 0] = t[0]
                gtcrop[joint, 1] = t[1]
                gtcrop[joint, 2] = gtorig[joint, 2]

            #print("{}".format(gt3Dorig))

            data.append(ImgFrame(dpt.astype(np.float32),gtorig,gtcrop,M,gt3Dorig,gt3Dcrop,com3D,dptFileName,'') )
            pbar.update(i)
            i+=1

            # early stop
            if len(data)>=Nmax:
                break

        pbar.finish()
        print("Loaded {} samples.".format(len(data)))

        if self.useCache:
            print("Save cache data to {}".format(pickleCache))
            f = open(pickleCache, 'wb')
            cPickle.dump((seqName,data,config), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

        # shuffle data
        if shuffle and rng is not None:
            print("Shuffling")
            rng.shuffle(data)
        return NamedImgSequence(seqName,data,config)

    def loadBaseline(self, filename, gt=None):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        if gt is not None:
            mat = scipy.io.loadmat(filename)
            names = mat['conv_joint_names'][0]
            joints = mat['pred_joint_uvconf'][0]

            self.numJoints = names.shape[0]

            data = []
            for dat in range(min(joints.shape[0], gt.shape[0])):
                fname = '{0:s}/depth_1_{1:07d}.png'.format(os.path.split(filename)[0], dat+1)
                if not os.path.isfile(fname):
                    continue
                dm = self.loadDepthMap(fname)
                # joints in image coordinates
                ev = np.zeros((self.numJoints, 3), np.float32)
                jt = 0
                for i in range(joints.shape[1]):
                    if np.count_nonzero(joints[dat, i, :]) == 0:
                        continue
                    ev[jt, 0] = joints[dat, i, 0]
                    ev[jt, 1] = joints[dat, i, 1]
                    ev[jt, 2] = dm[int(ev[jt, 1]), int(ev[jt, 0])]
                    jt += 1

                for jt in range(ev.shape[0]):
                    #if ev[jt,2] == 2001. or ev[jt,2] == 0.:
                    if abs(ev[jt, 2] - gt[dat, 13, 2]) > 150.:
                        ev[jt, 2] = gt[dat, jt, 2]#np.clip(ev[jt,2],gt[dat,13,2]-150.,gt[dat,13,2]+150.) # set to groundtruth if unknown

                ev3D = self.jointsImgTo3D(ev)
                data.append(ev3D)

            return data
        else:

            def nonblank_lines(f):
                for l in f:
                    line = l.rstrip()
                    if line:
                        yield line

            inputfile = open(filename)
            # first line specifies the number of 3D joints
            self.numJoints = len(inputfile.readline().split(' ')) / 3
            inputfile.seek(0)

            data = []
            for line in nonblank_lines(inputfile):
                part = line.split(' ')
                # joints in image coordinates
                ev = np.zeros((self.numJoints, 3), np.float32)
                for joint in range(ev.shape[0]):
                    for xyz in range(0, 3):
                        ev[joint, xyz] = part[joint*3+xyz]

                gt3Dworld = self.jointsImgTo3D(ev)

                data.append(gt3Dworld)

            return data

    def loadBaseline2D(self, filename):
        """
        Load baseline data
        :param filename: file name of data
        :return: list with joint coordinates
        """

        mat = scipy.io.loadmat(filename)
        names = mat['conv_joint_names'][0]
        joints = mat['pred_joint_uvconf'][0]

        self.numJoints = names.shape[0]

        data = []
        for dat in range(joints.shape[0]):
            # joints in image coordinates
            ev = np.zeros((self.numJoints,2),np.float32)
            jt = 0
            for i in range(joints.shape[1]):
                if np.count_nonzero(joints[dat,i,:]) == 0:
                    continue
                ev[jt,0] = joints[dat,i,0]
                ev[jt,1] = joints[dat,i,1]
                jt += 1

            data.append(ev)

        return data

    def jointsImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((sample.shape[0],3),np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.jointImgTo3D(sample[i])
        return ret

    def jointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f, see Thomson et al.
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (self.uy - sample[1]) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.joint3DToImg(sample[i])
        return ret

    def joint3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((3,),np.float32)
        #convert to metric using f, see Thomson et.al.
        if sample[2] == 0.:
            ret[0] = self.ux
            ret[1] = self.uy
            return ret
        ret[0] = sample[0]/sample[2]*self.fx+self.ux
        ret[1] = self.uy-sample[1]/sample[2]*self.fy
        ret[2] = sample[2]
        return ret

    def getCameraIntrinsics(self):
        """
        Get intrinsic camera matrix
        :return: 3x3 intrinsic camera matrix
        """
        ret = np.zeros((3, 3), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy  # TODO
        ret[2, 2] = 1
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        return ret

    def getCameraProjection(self):
        """
        Get homogenous camera projection matrix
        :return: 4x3 camera projection matrix
        """
        ret = np.zeros((4, 4), np.float32)
        ret[0, 0] = self.fx
        ret[1, 1] = -self.fy  # TODO
        ret[2, 2] = 1.
        ret[0, 2] = self.ux
        ret[1, 2] = self.uy
        ret[3, 2] = 1.
        return ret

    def showAnnotatedDepth(self, frame):
        """
        Show the depth image
        :param frame: image to show
        :return:
        """
        print("img min {}, max {}".format(frame.dpt.min(),frame.dpt.max()))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(frame.dpt, cmap=matplotlib.cm.jet, interpolation='nearest')
        ax.scatter(frame.gtcrop[:,0],frame.gtcrop[:,1])

        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[1::-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[1::-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[3:1:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[3:1:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[5:3:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[5:3:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[7:5:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[7:5:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[10:7:-1, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[10:7:-1, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[11, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[11, 1])), c='r')
        ax.plot(np.hstack((frame.gtcrop[13, 0], frame.gtcrop[12, 0])), np.hstack((frame.gtcrop[13, 1], frame.gtcrop[12, 1])), c='r')

        def format_coord(x, y):
            numrows, numcols = frame.dpt.shape
            col = int(x+0.5)
            row = int(y+0.5)
            if col>=0 and col<numcols and row>=0 and row<numrows:
                z = frame.dpt[row,col]
                return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
            else:
                return 'x=%1.4f, y=%1.4f' % (x, y)
        ax.format_coord = format_coord
        plt.show()
