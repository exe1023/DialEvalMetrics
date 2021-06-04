## Experiments

Uses [Pytorch Lightning](https://github.com/williamFalcon/pytorch-lightning) as the base framework.

## LM Finetuning

Clone pytorch-transformers first.

1. Load Prepare data for finetuning (`--only_data`)
2. Navigate to `pytorch-transformers/examples/lm_finetuning`
3. Pregenerate the data: 

  ```
  python pregenerate_training_data.py \
    --train_corpus ~/mlp/latentDialogAnalysis/fine_tune_convai2.txt \ --bert_model bert-base-uncased \ 
    --do_lower_case \
    --output_dir ~/mlp/latentDialogAnalysis/fine_tune_convai2_ep_10/ \ --epochs_to_generate 10 \
    --max_seq_len 256
  ```

4. Run finetuning script

  ```
  python finetune_on_pregenerated.py --pregenerated_data ~/mlp/latentDialogAnalysis/fine_tune_convai2_ep_10/ --bert_model bert-base-uncased --do_lower_case --output_dir ~/mlp/latentDialogAnalysis/fine_tune_convai2_ep_10_lm/ --epochs 10
  ```


## Extract model responses

```
python data.py
```

### Additional data files

- Responses used and outputs in the backtranslation, and the corrupted dialog files for Personachat data is [released here](https://drive.google.com/file/d/1NWTJNW-v7Y2yiFGcYVZi4H51Wsc3Bzqm/view?usp=sharing) (~1GB).

