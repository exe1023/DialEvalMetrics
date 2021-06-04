"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
from pytorch_lightning.logging import LightningLoggerBase, rank_zero_only
from logbook.logbook import LogBook

## TODO: do it!


class WandbLogger(LightningLoggerBase):
    def __init__(self, args):
        super().__init__()
        self.logbook = LogBook(vars(args))

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        # self.logbook = LogBook(params)
        self.logbook.write_metadata_logs(params)

    @rank_zero_only
    def log_metrics(self, metrics, step_num):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        metrics["minibatch"] = step_num
        self.logbook.write_metric_logs(metrics)

    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass
