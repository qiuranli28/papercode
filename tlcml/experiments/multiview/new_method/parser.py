# coding=utf-8
import argparse


class Parser(object):
    _parser = argparse.ArgumentParser("Usage")


    def __init__(self):
        super().__init__()
        self._setParseOpts()


    def parse(self):
        return self._parser.parse_args()


    def _setParseOpts(self):
        self._parser.add_argument("--dataset", dest="dataset", default="scene", type=str, help="dataset")
        self._parser.add_argument("--mode", dest="mode", default=1, type=int, help="mode")
        self._parser.add_argument("--batch_size", dest="batchSize", default=256, type=int, help="batch size")
        self._parser.add_argument("--epochs", dest="epochs", default=1000, type=int, help="epochs")
        self._parser.add_argument("--lr", dest="lr", default=1e-5, type=float, help="lr")
        self._parser.add_argument("--weightDecay", dest="weightDecay", default=1e-6, type=float, help="weightDecay")
        self._parser.add_argument("--annealing_step", dest="annealingStep", default=100, type=int, help="annealing step")
        self._parser.add_argument("--comment", dest="comment", default="", type=str, help="comment")
        self._parser.add_argument("--nonvisual", dest="nonvisual", default=True, action="store_false", help="is nonvisual")
        self._parser.add_argument("--seeds", dest="seeds", default=[0], type=int, nargs="+", help="seeds")
        self._parser.add_argument("--noise_degree", dest="noiseDegree", default="normal", type=str, help="noise degree")
        self._parser.add_argument("--subdir", dest="subdir", default="", type=str, help="tensorboard sub dir")

