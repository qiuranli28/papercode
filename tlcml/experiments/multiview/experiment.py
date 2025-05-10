# coding=utf-8
import os
import random
import numpy as np
import sklearn
import torch

from ..i_experiment import IExperiment
from ...utils.stdout.stdprint import *


class Experiment(IExperiment):
    def __init__(self):
        assert (not type(self) is Experiment), NotImplementedError
        super().__init__()

        self._seeds = []

    def fixSeed(self, seed=42):
        if (seed == None):
            return 1
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        return 0


    def _splitDataset(self, splitNum, ratio=(0.8, 0.2, 0)):

        rowLen = len(self._dataset)
        points = [0, int(rowLen * (ratio[0])), int(rowLen * (ratio[0] + ratio[1])), rowLen - 1]
        group = []
        for i in range(splitNum):
            np.random.seed(self._seeds[i % len(self._seeds)])
            index = np.arange(0, rowLen)
            np.random.shuffle(index)

            indexes = []
            for i in range(len(ratio)):
                indexes.append(index[points[i] : points[i + 1]].tolist())
            group.append(indexes)

        return group


    @DeprecationWarning
    def _splitDatasetWithValidset(self, splitNum, ratio=[0.6, 0.2, 0.2]):
        """
        splitNum: 划分的数量 == len(group)
        ratio: 训练集、测试集、验证集的比例
        """
        rowLen = len(self._dataset)
        points = [0, int(rowLen * (ratio[0])), int(rowLen * (ratio[0] + ratio[1])), rowLen - 1]
        group = []
        for i in range(splitNum):
            np.random.seed(self._seed + i)
            index = np.arange(0, rowLen)
            np.random.shuffle(index)

            indexes = []
            for i in range(len(ratio)):
                indexes.append(index[points[i] : points[i + 1]].tolist())
                # print(len(indexes[i]), type(indexes[i]), indexes)
            group.append(indexes)

        return group
