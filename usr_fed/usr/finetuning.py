import usr
import dr_api
import mlm_api

if __name__ == '__main__':
    drc_args, drf_args, mlm_args = usr.init_args()
    args = drc_args # just for extracting finetune task
    if args.finetune_task == 'mlm':
        args, model, tokenizer = mlm_api.init(mlm_args)
        mlm_api.train_main(args, model, tokenizer)
    elif args.finetune_task == 'drc':
        args, model, tokenizer = dr_api.init(drc_args)
        dr_api.train_main(args, model, tokenizer)
    elif args.finetune_task == 'drf':
        args, model, tokenizer = dr_api.init(drf_args)
        dr_api.train_main(args, model, tokenizer)
    else:
        raise Exception


