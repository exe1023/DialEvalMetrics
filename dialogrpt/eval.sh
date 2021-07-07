env CUDA_VISIBLE_DEVICES=1 python src/score.py test \
                        --max_n=-1 \
                          --data=test_data/${1}/data.tsv \
                          -p=restore/ensemble.yml       