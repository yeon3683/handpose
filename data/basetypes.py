from collections import namedtuple
ImgFrame = namedtuple('ImgFrame', ['dpt', 'gtorig', 'gtcrop', 'T', 'gt3Dorig', 'gt3Dcrop', 'com', 'fileName', 'subSeqName'])
NamedImgSequence = namedtuple('NamedImgSequence', ['name', 'data', 'config'])
