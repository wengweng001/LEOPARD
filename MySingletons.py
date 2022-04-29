# All of the code in this file was developed by Marcus de Carvalho without any modification

import torch
import gensim.downloader as gensim_downloader


class MyWord2Vec:
    def get(self):
        return Word2Vec.instance().word2vec

    def set(self, word2vec):
        Word2Vec.instance().word2vec = word2vec


class Word2Vec:
    class __Word2Vec:
        def __init__(self, word2vec=None):
            if word2vec:
                self.word2vec = word2vec
            else:
                print('Downloading (if needed) and setting Word2Vec Google-News-300 from gensim')
                print('This might take a while. Be patient...')
                self.word2vec = gensim_downloader.load('word2vec-google-news-300')
                print('Done!')

        def __str__(self):
            return repr(self) + self.word2vec

    _instance = None
    __instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls, word2vec=None):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            if word2vec is None:
                cls.__instance = Word2Vec.__Word2Vec()
            else:
                cls.__instance = Word2Vec.__Word2Vec(word2vec)
        return cls._instance

    def __getattr__(self, name):
        return getattr(self.__instance, name)


class MyDevice:
    def get(self):
        return TorchDevice.instance().device

    def set(self, is_gpu=True):
        if is_gpu:
            TorchDevice.instance().device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            TorchDevice.instance().device = torch.device("cpu")


class TorchDevice:
    class __TorchDevice:
        def __init__(self, device: torch.device = None):
            if device:
                self.device = device
            else:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def __str__(self):
            return repr(self) + self.device

    _instance = None
    __instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls, device: torch.device = None):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            if device is None:
                cls.__instance = TorchDevice.__TorchDevice()
            else:
                cls.__instance = TorchDevice.__TorchDevice(device)
        return cls._instance

    def __getattr__(self, name):
        return getattr(self.__instance, name)