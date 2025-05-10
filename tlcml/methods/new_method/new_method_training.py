# coding=utf-8
import numpy as np
from tqdm import tqdm
import torch
from copy import deepcopy

from ..utils.model_trainning import ModelTraining
from ...utils.stdout.stdprint import *


class NewMethodTraining(ModelTraining):
    def __init__(self, classNum=None, trainset=None, testset=None, validset=None, device=torch.device("cpu"), modelManager=None, isVisual=True):
        super().__init__(trainset, testset, validset, device, modelManager, isVisual=isVisual)
        self._classNum: None


    def train(self, epochs, annealingStep=500, gamma=0.5, patience=100):
        torch.set_printoptions(profile="full")
        model = self.getModel().to(self._device)
        bestLoss = np.inf
        bestAccu = 0
        epochWithoutImprovement = 0
        bestModelWeights = None
        step = 1
        optimizer = self._optimizerFunc(model.parameters(), **self._optimizerParamDict)
        for epoch in tqdm(range(epochs)):
            model.train()
            trainLoss = 0.0
            for data, target in self._trainloader:
                for key, curData in data.items():
                    data[key] = curData.to(self._device)
                target = target.to(self._device)
                output = model(data)
                loss = self._criterion(target, output, self._classNum, step, annealingStep=annealingStep, gamma=gamma, device=self._device)
                trainLoss += loss.detach().item()
                model.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
            trainLoss /= len(self._trainloader.dataset)
            self._modelVisualizer.addScalar("train_loss", trainLoss, step)
            if (self._validloader and len(self._validloader.dataset)):
                model.eval()
                validLoss = 0.0
                with torch.no_grad():
                    validAccu = 0
                    validCorrCount = 0
                    for batch_idx, (data, target) in enumerate(self._validloader):
                        for key, curData in data.items():
                            data[key] = curData.to(self._device)
                        target = target.to(self._device)
                        output = model(data)
                        loss = self._criterion(target, output, self._classNum, annealingStep, annealingStep=annealingStep, gamma=gamma, device=self._device)
                        validLoss += loss.item()
                        evidences, overallEvidence, fusedEvidences = output
                        overallAlpha = overallEvidence + 1
                        pred = overallAlpha.argmax(dim=1)
                        validCorrCount += (pred == target).sum().cpu().detach().numpy().item()
                    validAccu = validCorrCount / len(self._validset)
                    validLoss /= len(self._validloader.dataset)
                if (validLoss < bestLoss):
                    bestLoss = validLoss
                    bestAccu = validAccu
                    epochWithoutImprovement = 0
                    bestModelWeights = deepcopy(model).state_dict()
                    accu, totalEvicences, totalFusedEvidences = self.test(model)
                    PRINT_INFO(bestAccu, accu)
                else:
                    epochWithoutImprovement += 1
                if epochWithoutImprovement > patience:
                    model.load_state_dict(bestModelWeights)
                    accu, totalEvicences, totalFusedEvidences = self.test(model)
                    self._modelVisualizer.addScalar("test_accu", accu, epoch)
                    PRINT_INFO('Early stopping at epoch {}, bestAccu {}, accu {}...'.format(epoch + 1, bestAccu, accu))
                    break
            accu, totalEvicences, totalFusedEvidences = self.test(model)
            self._modelVisualizer.addScalar("test_accu", accu, epoch)
            if (epoch == epochs - 1):
                self.analyseEvidences(totalEvicences, totalFusedEvidences)
        PRINT_INFO("epoch: {}, train loss: {}, test accu: {}\n".format(epoch, trainLoss, accu), end="\n")

        torch.set_printoptions(profile="default")

    
    def test(self, model):
        totalEvicences = {key: np.empty((0, self._classNum)) for key in range(model.getViewNum())}
        totalFusedEvicences = {key: np.empty((0, self._classNum)) for key in range(model.getViewNum())}

        model.eval()
        correctCount = 0
        with torch.no_grad():
            for data, target in self._testloader:
                for key, curData in data.items():
                    data[key] = curData.to(self._device)
                target = target.to(self._device)
                evidences, overallEvidence, fusedEvidences = model(data)
                self.merge(totalEvicences, evidences)
                self.merge(totalFusedEvicences, fusedEvidences.transpose(0, 1))
                overallAlpha = overallEvidence + 1
                pred = overallAlpha.argmax(dim=1)
                correctCount += (pred == target).sum().cpu().detach().numpy().item()
            accu = correctCount / len(self._testset)

        return accu, totalEvicences, totalFusedEvicences


    def print(self, output, target):
        if (issubclass(type(output), dict)): return 0
        shape = output.shape
        shape = (shape[1], shape[0], shape[2])
        targetMetrix = torch.zeros(shape).to(output.device)
        for i in range(shape[0]):
            targetMetrix[i] = target[i]
        targetMetrix = targetMetrix.transpose(0, 1)
        PRINT_ERRORS("output", output, "target", targetMetrix)


    def merge(self, totalEvicences, evidences):
        for key, evidence in totalEvicences.items():
            totalEvicences[key] = np.concatenate((evidence, evidences[key].detach().cpu().numpy()), axis=0)


    def analyseEvidences(self, evidences, fusedEvidences):
        evidenceMatric = self.transformMatrix(evidences)
        fusedEvidenceMatric = self.transformMatrix(fusedEvidences)
        index = 10

    def transformMatrix(self, evidences):
        evidence0 = evidences[0]
        viewNum = len(evidences)
        batchSize, classNum = evidence0.shape
        evidenceMatric = np.zeros(shape=(viewNum, batchSize, classNum))
        index = 0
        for key, evidence in evidences.items():
            evidenceMatric[index] = evidences[key]
            index += 1
        evidenceMatric = evidenceMatric.transpose((1, 0, 2))
        return evidenceMatric

    def setClassNum(self, classNum):
        self._classNum = classNum

