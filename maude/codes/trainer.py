"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
# Lightning trainer
import torch
import numpy as np
import os
from time import sleep
from args import get_args
from logbook.logbook import LogBook
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger
from test_tube import Experiment, SlurmCluster
from codes.models import TransitionPredictorMaxPool
from codes.baseline_models import RuberUnreferenced, InferSent, BERTNLI
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from data import ParlAIExtractor
from addict import Dict

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def optimize_on_cluster(hyperparams):
    # enable cluster training
    # log all scripts to the test tube folder
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams, log_path=hyperparams.slurm_log_path,
    )

    # email for cluster coms
    # cluster.notify_job_status(email='add_email_here', on_done=True, on_fail=True)

    # configure cluster
    cluster.per_experiment_nb_gpus = hyperparams.per_experiment_nb_gpus
    cluster.per_experiment_nb_nodes = hyperparams.nb_gpu_nodes
    cluster.per_experiment_nb_cpus = hyperparams.per_experiment_nb_cpus
    cluster.job_time = hyperparams.job_time
    cluster.gpu_type = hyperparams.gpu_type
    cluster.memory_mb_per_node = 0

    # any modules for code to run in env
    cluster.add_command("source activate dialog")
    cluster.add_command(
        "export PYTHONPATH=$PYTHONPATH:/private/home/koustuvs/mlp/latentDialogAnalysis"
    )

    # run only on 32GB voltas
    # cluster.add_slurm_cmd(cmd='constraint', value='volta32gb',
    #                     comment='use 32gb gpus')
    cluster.add_slurm_cmd(
        cmd="partition", value=hyperparams.gpu_partition, comment="use 32gb gpus"
    )

    # run hopt
    # creates and submits jobs to slurm
    cluster.optimize_parallel_cluster_gpu(
        main,
        nb_trials=hyperparams.nb_hopt_trials,
        job_name=hyperparams.id + "_grid_search",
        job_display_name=hyperparams.id,
    )


def build_model(args):
    print("loading model...")
    logbook = LogBook(vars(args))
    logbook.write_metadata_logs(vars(args))
    # args = Dict(vars(args)) # fix for model saving error - maybe this is introducing the error
    if len(args.train_baseline) > 0 and args.train_baseline != "na":
        # load train data
        data = ParlAIExtractor(args, logbook)
        args.num_words = data.tokenizer.vocab_size
        if args.train_baseline == "ruber":
            model = RuberUnreferenced(args, logbook)
        elif args.train_baseline == "infersent":
            model = InferSent(args, logbook)
        elif args.train_baseline == "bertnli":
            model = BERTNLI(args, logbook)
    else:
        model = TransitionPredictorMaxPool(args, logbook)
    print("model built")
    return model


def main(args, cluster, results_dict=None):
    """
    Main training routine specific for this project
    :param hparams:
    :return:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = build_model(args)

    if args.use_cluster:
        # when using grid search, it's possible for all models to start at once
        # and use the same test tube experiment version
        relative_node_id = int(os.environ["SLURM_NODEID"])
        sleep(relative_node_id + 1)

    # ------------------------
    # 2 INIT TEST TUBE EXP
    # ------------------------

    # init logger
    # exp = Experiment(
    #     name=args.id,
    #     save_dir=args.model_save_dir,
    #     autosave=False,
    #     description='Dialog Eval'
    # )
    #

    # exp.argparse(args)
    # exp.save()

    # logger = WandbLogger(args)
    # logger = TestTubeLogger(
    #     save_dir=args.model_save_dir,
    #     name=args.id,
    #     debug=False,
    #     create_git_tag=False
    # )

    # ------------------------
    # 3 DEFINE CALLBACKS
    # ------------------------
    # model_save_path = '{}/{}/version_{}/weights'.format(args.model_save_dir, exp.name, exp.version)
    # early_stop = EarlyStopping(
    #     monitor='val_loss',
    #     patience=5,
    #     verbose=True,
    #     mode='min'
    # )
    #
    # ckpt_path = '{}/{}/version_{}/{}'.format(
    #     args.model_save_dir,
    #     args.id,
    #     logger.experiment.version,
    #     'weights')
    #
    # checkpoint = ModelCheckpoint(
    #     filepath=ckpt_path,
    #     save_best_only=True,
    #     verbose=True,
    #     monitor='val_loss',
    #     mode='min'
    # )

    if args.restore_version >= 0:
        logger = TestTubeLogger(
            save_dir=os.path.join(args.model_save_dir, args.id),
            version=args.restore_version,  # An existing version with a saved checkpoint
            name="lightning_logs",
        )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        # experiment=exp,
        logger=logger if args.restore_version >= 0 else True,
        default_save_path=os.path.join(args.model_save_dir, args.id),
        # checkpoint_callback=checkpoint,
        early_stop_callback=None,
        gpus="0" if args.debug else args.gpus,
        show_progress_bar=args.use_cluster == False,
        row_log_interval=10,
        log_save_interval=1,
        # cluster=cluster,
        # fast_dev_run=True,
        train_percent_check=0.001 if args.debug else args.train_per_check,
        val_percent_check=0.001 if args.debug else args.val_per_check,
        test_percent_check=0.001 if args.debug else args.test_per_check,
        distributed_backend="ddp"
        if args.use_ddp
        else "dp",  # if args.debug else 'ddp',  #  running on local gpus
        max_nb_epochs=20 if args.debug else 1000,
        # track_grad_norm=2,
        overfit_pct=0.001 if args.debug else 0.0,
        gradient_clip_val=args.clip,
        # amp_level='O2', use_amp=True,
        # overfit_pct=0.005
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)
    trainer.test()


if __name__ == "__main__":
    args = get_args()
    # Gen Experiment ID
    model = args.train_baseline if args.train_baseline else "transition"
    args.id = "{}_{}".format(model, args.corrupt_type)
    if args.use_cluster:
        args.batch_size = args.batch_size * args.per_experiment_nb_gpus
        optimize_on_cluster(args)
    else:
        main(args, None, None)
