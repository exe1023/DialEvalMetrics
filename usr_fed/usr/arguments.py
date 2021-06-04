import argparse
import torch

no_cuda = False if torch.cuda.is_available() else True
def init_args():
     parser = argparse.ArgumentParser()
     parser.add_argument('--finetune_task', default='mlm', type=str)
     # -----------   From train_understandable.py --------
     ## Required parameters
     parser.add_argument("--data_dir", default=None, type=str, required=False,
                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument("--model_type", default=None, type=str, required=False,
                         help="Model type selected in the list: " + ", ")
     parser.add_argument("--model_name_or_path", default=None, type=str, required=False,
                         help="Path to pre-trained model or shortcut name selected in the list: " + ", ")
     parser.add_argument("--task_name", default=None, type=str, required=False,
                         help="The name of the task to train selected in the list: " + ", ")
     parser.add_argument("--output_dir", default=None, type=str, required=False,
                         help="The output directory where the model predictions and checkpoints will be written.")
     parser.add_argument("--pretrained_dir", default=None, type=str, required=False,
                         help="The output directory where the model predictions and checkpoints will be written.")

     ## Other parameters
     parser.add_argument("--config_name", default="", type=str,
                         help="Pretrained config name or path if not the same as model_name")
     parser.add_argument("--tokenizer_name", default="", type=str,
                         help="Pretrained tokenizer name or path if not the same as model_name")
     parser.add_argument("--cache_dir", default="", type=str,
                         help="Where do you want to store the pre-trained models downloaded from s3")
     parser.add_argument("--max_seq_length", default=128, type=int,
                         help="The maximum total input sequence length after tokenization. Sequences longer "
                              "than this will be truncated, sequences shorter will be padded.")
     parser.add_argument("--do_train", action='store_true',
                         help="Whether to run training.")
     parser.add_argument("--do_eval", action='store_true',
                         help="Whether to run eval on the dev set.")
     parser.add_argument("--evaluate_during_training", action='store_true',
                         help="Rul evaluation during training at each logging step.")
     parser.add_argument("--do_lower_case", action='store_true',
                         help="Set this flag if you are using an uncased model.")

     parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                         help="Batch size per GPU/CPU for training.")
     parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                         help="Batch size per GPU/CPU for evaluation.")
     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                         help="Number of updates steps to accumulate before performing a backward/update pass.")
     parser.add_argument("--learning_rate", default=5e-5, type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--weight_decay", default=0.0, type=float,
                         help="Weight deay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                         help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     parser.add_argument("--num_train_epochs", default=3.0, type=float,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--max_steps", default=-1, type=int,
                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
     parser.add_argument("--warmup_steps", default=0, type=int,
                         help="Linear warmup over warmup_steps.")

     parser.add_argument('--logging_steps', type=int, default=50,
                         help="Log every X updates steps.")
     parser.add_argument('--save_steps', type=int, default=50,
                         help="Save checkpoint every X updates steps.")
     parser.add_argument("--eval_all_checkpoints", action='store_true',
                         help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
     parser.add_argument("--no_cuda", action='store_true',
                         help="Avoid using CUDA when available")
     parser.add_argument('--overwrite_output_dir', action='store_true',
                         help="Overwrite the content of the output directory")
     parser.add_argument('--overwrite_cache', action='store_true',
                         help="Overwrite the cached training and evaluation sets")
     parser.add_argument('--no_cache', action='store_true',
                         help='Do not cache the feature file')
     parser.add_argument('--seed', type=int, default=42,
                         help="random seed for initialization")

     parser.add_argument('--tpu', action='store_true',
                         help="Whether to run on the TPU defined in the environment variables")
     parser.add_argument('--tpu_ip_address', type=str, default='',
                         help="TPU IP address if none are set in the environment variables")
     parser.add_argument('--tpu_name', type=str, default='',
                         help="TPU name if none are set in the environment variables")
     parser.add_argument('--xrt_tpu_config', type=str, default='',
                         help="XRT TPU config if none are set in the environment variables")

     parser.add_argument('--fp16', action='store_true',
                         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
     parser.add_argument('--fp16_opt_level', type=str, default='O1',
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html")
     parser.add_argument("--local_rank", type=int, default=-1,
                         help="For distributed training: local_rank")
     parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
     parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

     # -----------   Plus unique arguments in run_lm_fintuning.py --------
     parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a text file).")
     parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

     parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
     parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

     parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
     parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')


     args = parser.parse_args()
     return args 

def arc_args(model_path):
     args = init_args()
     args.per_gpu_eval_batch_size = 1
     args.pretrained_dir = model_path
     if args.output_dir is None:
          args.output_dir = model_path
     args.model_type = 'roberta'
     args.model_name_or_path = 'roberta-base'
     args.do_eval = True
     args.task_name = 'qqp'
     args.no_cache = True
     args.no_cuda = no_cuda
     return args

def arf_args(model_path):
     args = init_args()
     args.per_gpu_eval_batch_size = 1
     args.pretrained_dir = model_path
     if args.output_dir is None:
          args.output_dir = model_path
     args.model_type = 'roberta'
     args.model_name_or_path = 'roberta-base'
     args.do_eval = True
     args.task_name = 'qqp'
     args.no_cache = True
     args.no_cuda = no_cuda
     return args

def mlm_args(model_path):
     args = init_args()
     args.per_gpu_eval_batch_size = 1
     args.pretrained_dir = model_path
     if args.output_dir is None:
          args.output_dir = model_path
     args.model_type = 'roberta'
     args.model_name_or_path = 'roberta-base'
     args.do_eval = True
     args.mlm = True
     args.no_cache = True
     args.no_cuda = no_cuda
     return args