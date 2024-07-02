firectl create fine-tuning-job --settings-file /Users/jameschua/ml/cot-transparency/scripts/colm_grid_experiments/firectl_yamls/llama_10k_70b.yaml --display-name "llama-70b-suggested"
firectl create deployment llama-v3-70b-instruct --min-replica-count 0 --max-replica-count 3
