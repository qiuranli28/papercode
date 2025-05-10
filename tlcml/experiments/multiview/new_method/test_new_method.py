# coding=utf-8
import os
import json
import random
from enum import Enum
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from sklearn.model_selection import KFold

from ....methods.new_method.new_method import NewMethod
from ....methods.lenet.lenet import Lenet
from ....methods.new_method.evidence_net import EvidenceNet
from ....utils.stdout.stdprint import *
from ....utils.data_handlers.dataset.multiview.dataset import Dataset as MultiviewDataset
from ....methods.utils.dataset_info import DatasetInfo
from ..experiment import Experiment
from ....methods.utils.functional.logits_to_evidence import LogitsToEvidence
from .parser import Parser


class TestNewMethod(Experiment):
    def __init__(self, logitToEvidenceFunc : list=[LogitsToEvidence.getExpEvidence()], seeds=[]):
        super().__init__()
        self._normalize = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._subnets = []
        self._subnetArgs = []
        self._subnetKwargs = []
        self._dataset = None
        self._method: Method
        self._seeds = seeds if (seeds) else [0]
        self._NnMudule: torch.nn.Module
        self._subnetArgs: list
        self._subnetKwargs: dict
        self._logitToEvidenceFunc = logitToEvidenceFunc
        self._viewNum = 1
        self._classNum = 1
        self._virtualViewNum = self._viewNum


    def loadDataset(self, path, normalize, viewNum=0):
        assert (os.path.exists(path)), AssertionError
        self._normalize = normalize
        dataset = np.load(path, allow_pickle=True).item()
        data = dataset["data"]
        if (viewNum > 0):
            data = {}
            for k, v in dataset["data"].items():
                if (len(data) >= viewNum): break
                data[k] = v
        PRINT_INFO("view number: {}".format(len(data)))
        self._dataset = MultiviewDataset(data, dataset["target"], normalize=self._normalize)


    def getDataDim(self):
        return self._dataset.getDataDims()


    def run(self, epochs=500, batchSize=256, lr=3e-3, weightDecay=1e-5, splitNum=5, shuffle=True, annealingStep=50, subdir="new_method", comment="", noiseDegree = "normal"):
        assert (self._dataset)
        # self.fixSeed(self._seeds)
        group = self._splitDataset(splitNum)
        # group = self._splitDatasetWithValidset(splitNum)
        for index, (trainIndex, testIndex, validIndex) in enumerate(group):
            if (index >= len(self._seeds)): break
            seed = self._seeds[index % len(self._seeds)]
            self.fixSeed(seed)
            trainData, trainTarget = self._dataset[trainIndex]
            trainset = MultiviewDataset(trainData, trainTarget, normalize=self._normalize)
            testData, testTarget = self._dataset[testIndex]
            testset = MultiviewDataset(testData, testTarget, normalize=self._normalize)
            if (validIndex):
                validData, validTarget = self._dataset[validIndex]
                validset = MultiviewDataset(validData, validTarget, normalize=self._normalize)
            else:
                validset = None

            # 添加噪音
            with open("./src/experiments/multiview/noise.json", 'r') as file:
                datainfo = json.load(file)
            info = datainfo[noiseDegree]
            sigma = float(info["sigma"])
            noiseRatio = float(info["noiseRatio"])
            conflictRatio = float(info["conflictRatio"])

            if (noiseRatio > 0 or conflictRatio > 0):
                testset.postprocessing(addNoise=True, sigma=sigma, noiseRatio=noiseRatio, addConflict=True, conflictRatio=conflictRatio)
                if (validset): validset.postprocessing(addNoise=True, sigma=sigma, noiseRatio=noiseRatio, addConflict=True, conflictRatio=conflictRatio)
                curcomment = comment + "_conflict:{}:{}:{}".format(sigma, noiseRatio, conflictRatio)
            else:
                curcomment = comment + "_normal"

            PRINT_INFO("comment", curcomment)
            self._updateSubnets()
            self._method.createVisualizer(subdir="/".join([subdir, args.subdir if args.subdir else "realtime"]), comment=curcomment + "_{}_{}".format(index, seed))
            self._method.setDatasets(trainset, testset, validset)
            self._method.configModelTraining(batchSize=batchSize, lr=lr, weightDecay=weightDecay, shuffle=shuffle)
            self._method.train(epochs=epochs, annealingStep=annealingStep)


    def configNetParams(self, NnModule, args=[], kwargs=[]):
        self._NnMudule = NnModule
        self._subnetArgs = args if (args) else [[] for i in range(self._virtualViewNum)]
        self._subnetKwargs = kwargs if (kwargs) else [dict() for i in range(self._virtualViewNum)]


    def _updateSubnets(self):
        self._subnets.clear()
        for i in range(self._viewNum):
            funcs = self._logitToEvidenceFunc[i % len(self._logitToEvidenceFunc)]
            if (hasattr(funcs, "__call__")):
                funcs = [funcs]
            for j in range(len(funcs)):
                self._subnets.append(self._NnMudule(*(self._subnetArgs[i]), **(self._subnetKwargs[i])))
        self._method.configNet(self._subnets, logitToEvidenceFunc=self._logitToEvidenceFunc)


    def __call__(self, dsKind, epochs=500, batchSize=200, lr=1e-5, weightDecay=1e-5, splitNum=5, shuffle=True, annealingStep=50, comment="", isVisual=True, noiseDegree="normal"):
        methodname = "fusion"
        with open("./src/experiments/multiview/data.json", 'r') as file:
            datainfo = json.load(file)
        info = datainfo[dsKind]
        datasetPath = info["path"]
        isnormal = info["normal"]
        comment = "{}".format(info["comment"]) + ("_{}".format(comment) if (comment) else comment)

        self.loadDataset(datasetPath, isnormal)
        self._viewNum, self._classNum = self._dataset.getDatasetInfo()
        self._virtualViewNum = self._getVirtualViewNum(self._viewNum, self._logitToEvidenceFunc)
        # exit(0)
        comment += "_{}".format(methodname)
        dataDim = self.getDataDim()
        self._method = NewMethod(viewNum=self._virtualViewNum, classNum=self._classNum, device=self.device, isVisual=isVisual, realViewNum=self._viewNum)
        self.configNetParams(EvidenceNet, [[dim, self._classNum] for dim in dataDim])

        self.run(epochs=epochs, batchSize=batchSize, lr=lr, weightDecay=weightDecay, splitNum=splitNum, shuffle=shuffle, annealingStep=annealingStep, comment=comment, noiseDegree=noiseDegree)

    def _getVirtualViewNum(self, viewNum, funcs):
        funcLen = len(self._logitToEvidenceFunc)
        virtualViewNum = 0
        for i in range(self._viewNum):
            idx = i % funcLen
            func = self._logitToEvidenceFunc[idx]
            if (hasattr(func, "__call__")):
                virtualViewNum += 1
            else:
                virtualViewNum += len(func)
        return virtualViewNum


if __name__ == "__main__":
    global args
    args = Parser().parse()
    PRINT_INFO(args)
    seeds = args.seeds
    demo = TestNewMethod(logitToEvidenceFunc=[[LogitsToEvidence.getExpEvidence()]], seeds=seeds)
    demo(args.dataset, epochs=args.epochs, lr=args.lr, weightDecay=args.weightDecay, batchSize=args.batchSize, annealingStep=args.annealingStep, comment=args.comment, isVisual=args.nonvisual, noiseDegree=args.noiseDegree)
