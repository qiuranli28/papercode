# coding=utf-8
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.nn.functional import cosine_similarity

from ..lenet.lenet import Lenet
from ..utils.functional.logits_to_evidence import LogitsToEvidence
from ...utils.transform.one_hot_transfer import OneHotTransfer
from ...utils.stdout.stdprint import *

from ..method import Method
from .new_method_net import NewMethodNet
from .new_method_training import NewMethodTraining
from ...methods.utils.functional.evidence_loss_functional import EvidenceLossFunctional

class NewMethod(Method):
    def __init__(self, viewNum, classNum, isVisual=True, realViewNum=0, device=torch.device("cpu")):
        super().__init__()
        self._viewNum: int = viewNum
        self._classNum = classNum if (classNum > 1) else 1
        self._device = device
        self._model = NewMethodNet(viewNum, classNum, realViewNum=realViewNum, device=device)
        self._modelTraining = NewMethodTraining(device=device, isVisual=isVisual)


    def configNet(self, subnets: list, logitToEvidenceFunc=LogitsToEvidence.getSoftplusEvidence()):
        self._model.setSubnets(subnets)
        self._model.setLogitToEvidenceFunc(logitToEvidenceFunc)


    def configModelTraining(self, batchSize=1000, lr=0.003, weightDecay=1e-5, shuffle=True):
        assert ((self._trainset is not None) and (self._testset is not None)), ValueError
        criterion = NewMethod.lossFunc
        self._modelTraining.setClassNum(self._classNum)
        self._modelTraining.setDataset(self._trainset, self._testset, self._validset)
        self._modelTraining.makeDataLoaders(batchSize, shuffle=shuffle)
        self._modelTraining.setModel(self._model)
        self._modelTraining.setCriterion(criterion)
        self._modelTraining.setOptimizer(torch.optim.Adam, lr=lr, weight_decay=weightDecay)


    def train(self, epochs=300, annealingStep=50):
        ans = self._modelTraining.train(epochs=epochs, annealingStep=annealingStep)


    def addDatasetNoise(self, ratio, **kwNoiseArgs):
        """
        scale: 噪声污染view的比例
        kwNoiseArgs: 噪音参数
        """
        from ...utils.data_handlers.noise.gaussian_noise_feeder import GaussianNoiseFeeder
        noiseTransform = GaussianNoiseFeeder(**kwNoiseArgs)
        dataset = self._testset
        count = len(dataset)
        for i in range(count):
            viewData, target = dataset[i]
            viewKeys = viewData.keys()
            index = np.random.randint(0, self._viewNum)
            if (np.random.rand() < ratio):
                dataset[i][0][index] = noiseTransform(viewData[index])
                print(np.sum(viewData[index] != dataset[i][0][index]))


    @staticmethod
    def lossFunc(target, output, classNum, step, annealingStep, gamma, device):
        target = OneHotTransfer(classNum)(target)
        evidences, overallEvidence, fusedEvidences = output
        overallAlpha = overallEvidence + 1

        accu = NewMethod.crossEntropyAndKlLoss(target, overallAlpha, step, annealingStep)
        viewNum = len(evidences)
        for view in range(viewNum):
            evidence = evidences[view]
            alpha = evidence + 1
            accu += NewMethod.crossEntropyAndKlLoss(target, alpha, step, annealingStep)
            uncertainty = alpha.shape[1] / alpha.sum(dim=1)
        accu = accu / (viewNum + 1)
        loss = accu
        loss += gamma * NewMethod.getConsistencyLossV3(fusedEvidences, device)
        loss += NewMethod.getSelfInspectionLoss(fusedEvidences, evidences)

        return loss


    @staticmethod
    def getSelfInspectionLoss(output, target):
        if (issubclass(type(output), dict)): return 0
        shape = output.shape
        shape = (shape[1], shape[0], shape[2])
        targetMatrix = torch.zeros(shape).to(output.device)
        for i in range(shape[0]):
            targetMatrix[i] = target[i]
        targetMatrix = targetMatrix.transpose(0, 1)
        mseLoss = torch.nn.MSELoss()(torch.log(output.sum(dim=2).sum(dim=1)), torch.log(targetMatrix.sum(dim=2).sum(dim=1)))
        return mseLoss


    @staticmethod
    def crossEntropyAndKlLoss(target, alpha, step, annealingStep):
        item1 = EvidenceLossFunctional.getCrossEntropyLoss(target, alpha)
        item2 = EvidenceLossFunctional.getKlDivergence(target, alpha, step, annealingStep)
        return torch.mean(item1 + item2)


    @staticmethod
    def getConsistencyLossV3(evidences, device):
        if (isinstance(evidences, dict)):
            evidence0 = evidences[0]
            viewNum = len(evidences)
            batchSize, classNum = evidence0.shape
            evidenceMetric = torch.zeros(size=(viewNum, batchSize, classNum)).to(device)
            index = 0
            for key, evidence in evidences.items():
                evidenceMetric[index] = evidences[key]
                index += 1
        else:
            evidenceMetric = evidences.transpose(0, 1)
            viewNum, batchSize, classNum = evidenceMetric.shape
        alphaMetric = evidenceMetric + 1

        alphaSumMetric = torch.sum(alphaMetric, dim=2, keepdim=True)
        probabilityMetric = alphaMetric / alphaSumMetric
        uncertaintyMetric = classNum / alphaSumMetric
        confidenceMetric = 1 / uncertaintyMetric
        consistencyMetric = torch.zeros(size=(batchSize,)).to(device)
        for view in range(viewNum):
            viewProb = probabilityMetric[view]
            viewConfidence = confidenceMetric[view]
            diffMetric = torch.pow((probabilityMetric - viewProb), exponent=2)
            projectedDistance = torch.sum(diffMetric, dim=2, keepdim=True) / 2
            conjectiveCertaintyMetric = confidenceMetric + viewConfidence
            viewConsistencyMetric = conjectiveCertaintyMetric * projectedDistance
            viewConsistencyMetric = torch.sum(viewConsistencyMetric, dim=0, keepdim=True)
            viewConsistencyMetric = torch.squeeze(viewConsistencyMetric, dim=(0, 2))
            consistencyMetric += viewConsistencyMetric

        consistencyMetric /= viewNum - 1
        return torch.mean(consistencyMetric)

    @staticmethod
    def getCrossEntropyLoss(target, alpha):
        loss = target - alpha
        loss = torch.pow(loss, 2)
        loss = loss.sum(dim = 1) / loss.shape[1]
        return loss.mean()

