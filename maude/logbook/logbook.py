"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
"""Wrapper over wandb api"""

import wandb
import collections
from logbook import filesystem_logger as fs_log


def flatten_dict(d, parent_key="", sep="#"):
    """Method to flatten a given dict using the given seperator.
    Taken from https://stackoverflow.com/a/6027615/1353861
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class LogBook:
    """Wrapper over comet_ml api"""

    def __init__(self, config):
        self._experiment_id = 0
        self.metrics_to_record = [
            # "mode ",
            # "num_timesteps",
            # "mode",
            # "minibatch",
            # "loss",
            # "true_loss",
            # "false_loss",
            # "score_mean",
            # "score_std",
            # "scramble_score_mean",
            # "scramble_score_std",
            # "mean_diff",
            # "epoch"
        ]

        flattened_config = flatten_dict(config, sep="_")

        self.should_use_remote_logger = config["remote_logging"]

        if self.should_use_remote_logger:
            wandb.init(
                config=flattened_config,
                project=config["wandb_project"],
                name=config["id"],
                dir=config["logger_dir"],
            )

        self.should_use_tb = False

        fs_log.set_logger(config)

    def _log_metrics(self, dic, prefix, step):
        """Method to log metric"""
        formatted_dict = {}
        for key, val in dic.items():
            formatted_dict[prefix + "_" + key] = val
        if self.should_use_remote_logger:
            wandb.log(formatted_dict, step=step)

    def write_config_log(self, config):
        """Write config"""
        fs_log.write_config_log(config)
        flatten_config = flatten_dict(config, sep="_")
        flatten_config["experiment_id"] = self._experiment_id

    def write_metric_logs(self, metrics):
        """Write Metric"""
        metrics["experiment_id"] = self._experiment_id
        fs_log.write_metric_logs(metrics)
        flattened_metrics = flatten_dict(metrics, sep="_")

        if self.metrics_to_record:
            metric_dict = {
                key: flattened_metrics[key]
                for key in self.metrics_to_record
                if key in flattened_metrics
            }
        else:
            metric_dict = flattened_metrics
        prefix = metrics.get("mode", None)
        num_timesteps = metric_dict.pop("minibatch")
        self._log_metrics(dic=metric_dict, prefix=prefix, step=num_timesteps)

        if self.should_use_tb:

            timestep_key = "num_timesteps"
            for key in set(list(metrics.keys())) - set([timestep_key]):
                self.tensorboard_writer.add_scalar(
                    tag=key,
                    scalar_value=metrics[key],
                    global_step=metrics[timestep_key],
                )

    def write_compute_logs(self, **kwargs):
        """Write Compute Logs"""
        kwargs["experiment_id"] = self._experiment_id
        fs_log.write_metric_logs(**kwargs)
        metric_dict = flatten_dict(kwargs, sep="_")

        num_timesteps = metric_dict.pop("num_timesteps")
        self._log_metrics(dic=metric_dict, step=num_timesteps, prefix="compute")

    def write_message_logs(self, message):
        """Write message logs"""
        fs_log.write_message_logs(message, experiment_id=self._experiment_id)

    def write_metadata_logs(self, metadata):
        """Write metadata"""
        metadata["experiment_id"] = self._experiment_id
        fs_log.write_metadata_logs(metadata)
        # self.log_other(key="best_epoch_index", value=kwargs["best_epoch_index"])

    def watch_model(self, model):
        """Method to track the gradients of the model"""
        wandb.watch(models=model, log="all")
