import torch
import torch.nn as nn
import torchvision
from ...utils.stdout.stdprint import *

class FusionNet(nn.Module):
    def __init__(self, viewNum, classNum, device):
        super().__init__()

        self._viewNum = viewNum
        self._classNum = classNum
        self._device = device
        self._dropoutProb = 0.2
        self._eps = 1e-5

        self.__initModules()


    def forward(self, x):
        return self.getTwoLayerForward(x)

    def getTwoLayerForward(self, x):
        origin = x
        rowX = self.rowConv(x.transpose(1, 2))
        colX = self.colConv(x)
        rowX = self.flatten(rowX)
        colX = self.flatten(colX)
        # layer 1
        rowX = self.linear11(rowX)
        rowX = self._norm1d11(rowX)
        rowX = self.relu(rowX)
        rowX = self.dropout(rowX)
        colX = self.linear12(colX)
        colX = self._norm1d12(colX)
        colX = self.relu(colX)
        colX = self.dropout(colX)
        # layer 2
        x = torch.cat([rowX, colX], dim=1)
        x = self.linear2(x)
        x = self._norm1d(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.reshape(-1, self._viewNum, self._classNum)
        x = origin * x
        evidences = x + self._eps
        overallEvidence = torch.mean(evidences, dim=1)
        return evidences, overallEvidence

    def __initModules(self):
        # 行卷积
        rowInChannel = self._classNum
        rowOutChannel = 32
        self.rowConv = nn.Conv1d(rowInChannel, rowOutChannel, kernel_size=1).to(self._device)
        # 列卷积
        colInChannel = self._viewNum
        colOutChannel = 32
        self.colConv = nn.Conv1d(colInChannel, colOutChannel, kernel_size=1).to(self._device)
        self.flatten = nn.Flatten().to(self._device)
        # linear
        inChannel = rowInChannel * rowOutChannel + colInChannel * colOutChannel
        outChannel = self._viewNum * self._classNum
        self.linear = nn.Linear(inChannel, outChannel).to(self._device)
        # relu
        self.relu = nn.ReLU(False).to(self._device)
        # dropout
        self.dropout = nn.Dropout(self._dropoutProb).to(self._device)
        # resize
        self._norm1d = nn.BatchNorm1d(num_features=outChannel, eps=1e-5)
        self.contribution = nn.Parameter(torch.ones(size=(self._viewNum, 1), dtype=torch.float32), requires_grad=True)
        # two layer
        rowCount = colInChannel * rowOutChannel
        self.linear11 = nn.Linear(rowCount, rowCount // 2).to(self._device)
        colCount = rowInChannel * colOutChannel
        self.linear12 = nn.Linear(colCount, colCount // 2).to(self._device)
        self.linear2 = nn.Linear(rowCount // 2 + colCount // 2, outChannel).to(self._device)
        self._norm1d11 = nn.BatchNorm1d(rowCount // 2).to(self._device)
        self._norm1d12 = nn.BatchNorm1d(colCount // 2).to(self._device)

    



