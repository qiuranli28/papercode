# coding=utf-8

class IExperiment(object):
    def __init__(self):
        assert (not type(self) is IExperiment), NotImplementedError


    def loadDataset(self, path):
        raise NotImplementedError


    def run(self, splitNum, shuffle):
        raise NotImplementedError


    def _splitDataset(self, splitNum, shuffle):
        raise NotImplementedError
