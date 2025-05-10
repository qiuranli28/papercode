# coding=utf-8
import copy 
import torch
import torch.nn as nn
from ...utils.stdout.stdprint import *
from .fusion_net import FusionNet
from .attention_net import AttentionNet
from ..initialization.model_initialization import ModelInitialization

class NewMethodNet(nn.Module):
    def __init__(self, viewNum, classNum, subnets: list=[], logitToEvidenceFunc=None, device=torch.device("cpu"), realViewNum=0):
        super().__init__()
        self.subnets = nn.ModuleList([])
        self.fusionNet = None
        self.attentionNet = None

        self.setViewNum(viewNum)
        self.setClassNum(classNum)
        self.setLogitToEvidenceFunc(logitToEvidenceFunc)
        self.setSubnets(subnets)
        self._device = device

        self._threshod = 1e-5
        self._logitToEvidenceFunc = []
        self._realViewNum = realViewNum if (realViewNum) else viewNum


    def forward(self, x):
        return self.getNetFusion(x)


    def setViewNum(self, count: int):
        self._viewNum = count if (count >= 1) else 1


    def setClassNum(self, classNum):
        self._classNum = classNum if (classNum >= 1) else 1


    def getViewNum(self):
        return self._viewNum


    def setSubnets(self, subnets: list):
        assert (isinstance(subnets, list)), TypeError
        if (not subnets): return
        self.subnets = nn.ModuleList([])
        for i in range(len(subnets)):
            self.subnets.append(subnets[i].to(self._device))

        self.fusionNet = FusionNet(self._viewNum, self._classNum, self._device).to(self._device)
        self.attentionNet = AttentionNet(self._classNum, 5, self._classNum).to(self._device)


    def setLogitToEvidenceFunc(self, funcs):
        if (isinstance(funcs, list)):
            self._logitToEvidenceFunc = funcs
            while (len(self._logitToEvidenceFunc) < self._viewNum):
                self._logitToEvidenceFunc += funcs
            self._logitToEvidenceFunc = self._logitToEvidenceFunc[0 : self._viewNum]
        else:
            self._logitToEvidenceFunc = [funcs for i in range(self._viewNum)]


    def getMeanFusion(self, x):
        evidences = dict()
        batchSize = x[0].shape[0]
        shape = (self._viewNum, batchSize, self._classNum)
        evidenceMatrix = torch.zeros(shape).to(self._device)
        for view in range(self._viewNum):
            data = x[view]
            output = self.subnets[view](data)
            evidence = self._logitToEvidenceFunc[view](output)
            evidences[view] = evidence
            evidenceMatrix[view] = evidence
        fusionEvidence = evidenceMatrix.mean(dim=0)
        fusedEvidences = evidenceMatrix.transpose(0, 1)
        return evidences, fusionEvidence, fusedEvidences


    def getProcessedAddtion(self, x):
        evidences = dict()
        shape = (x[0].shape[0], self._classNum)
        overallEvidence = torch.zeros(shape).to(self._device)
        for view in range(self._viewNum):
            data = x[view]
            output = self.subnets[view](data)
            evidence = self._logitToEvidenceFunc[view](output)
            evidences[view] = evidence
            weight = 1
            if (not self.training):
                mean = evidence.mean(dim=1)
                weight = 0.1
                availIndex = (mean < self._threshod)
                index = torch.arange(0, evidence.shape[0]).to(self._device)
                for i in availIndex:
                    evidence[i] = overallEvidence[i]
            overallEvidence = (weight * evidence + overallEvidence) / 2
        return (evidences, overallEvidence)


    def getNetFusion(self, x):
        evidences = dict()
        batchSize = x[0].shape[0]
        shape = (self._viewNum, batchSize, self._classNum)
        evidenceMatrix = torch.zeros(shape).to(self._device)
        virtualView = 0
        funcLen = len(self._logitToEvidenceFunc)
        for view in range(self._realViewNum):
            data = x[view]
            output = self.subnets[virtualView](data)
            logitFunc = self._logitToEvidenceFunc[view % funcLen]
            if (hasattr(logitFunc, "__call__")):
                logitFunc = [logitFunc]
            for func in logitFunc:
                evidence = func(output)
                evidences[virtualView] = evidence
                evidenceMatrix[virtualView] = evidence
                virtualView += 1

        evidenceMatrix = evidenceMatrix.transpose(0, 1)
        fusedEvidences, fusionEvidence = self.fusionNet(evidenceMatrix)
        return evidences, fusionEvidence, fusedEvidences



