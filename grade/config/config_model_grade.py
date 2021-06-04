
MASK_BY_SHEDULE = False

name = "GRADE"
hidden_size = 768
dropout = 0.1
bert_vocab_size = 30522
dailydialog_vocab_size = 20202

dim_c_768 = 768
dim_c_300 = 300
dim_c_512 = 512
dim_e = 1
num_units_300 = 300


word_embedder_300 = {
    'dim': num_units_300,
    "dropout_rate": 0,
    "dropout_strategy": 'element',
    "name": "word_embedder"
}


bert_encoder = {
    'pretrained_model_name': 'bert-base-uncased',
    'embed': {
        'dim': 768,
        'name': 'word_embeddings'
    },
    'vocab_size': bert_vocab_size,
    'segment_embed': {
        'dim': 768,
        'name': 'token_type_embeddings'
    },
    'type_vocab_size': 2,
    'position_embed': {
        'dim': 768,
        'name': 'position_embeddings'
    },
    'position_size': 512,

    'encoder': {
        'dim': 768,
        'embedding_dropout': 0.1,
        'multihead_attention': {
            'dropout_rate': 0.1,
            'name': 'self',
            'num_heads': 12,
            'num_units': 768,
            'output_dim': 768,
            'use_bias': True
        },
        'name': 'encoder',
        'num_blocks': 12,
        'poswise_feedforward': {
            'layers': [
                {
                    'kwargs': {
                        'in_features': 768,
                        'out_features': 3072,
                        'bias': True
                    },
                    'type': 'Linear'
                },
                {"type": "BertGELU"},
                {
                    'kwargs': {
                        'in_features': 3072,
                        'out_features': 768,
                        'bias': True
                    },
                    'type': 'Linear'
                }
            ]
        },
        'residual_dropout': 0.1,
        'use_bert_config': True
    },
    'hidden_size': 768,
    'initializer': None,
    'name': 'bert_encoder',
}


