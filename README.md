# Shakespeak

Toy Transformer implementation trained on character level tokens on Shakesperian text.
Part of the Deep Learning Course at the University of Geneva.

### Launching a cross-validation run

```
python3 ./train.py --batch_size=12 --n_tokens=64 --n_layers=2 --n_heads=2 --d_model=32 --use_lr_decay=True --dataset_path='./datasets/shakespear_corpus.txt' --max_iter=200 --val_int=25 --cross_val=True --k_fold=10 --save=True --save_int=50 --name=YvanTest
```


### Unit Tests

From Bash, run 
```bash
for f in ./test/*.py; do python3 "$f"; done
```