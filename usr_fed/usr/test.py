import usr
if __name__ == '__main__':
    with open('dialogue_context.txt') as f:
        context = f.readlines()
    with open('dialogue_response.txt') as f:
        response = f.readlines()
    drc_args, drf_args, mlm_args = usr.init_args()
    drc_args, drc_model, drc_tokenizer, \
    drf_args, drf_model, drf_tokenizer, \
    mlm_args, mlm_model, mlm_tokenizer = usr.init_models(drc_args, drf_args, mlm_args)

    usr.get_scores(context, response, 
                   drc_args, drc_model, drc_tokenizer,
                   drf_args, drf_model, drf_tokenizer,
                   mlm_args, mlm_model, mlm_tokenizer)
