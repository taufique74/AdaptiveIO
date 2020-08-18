# AdaptiveIO [WIP]
Language modeling with Adaptive Softmax and Adaptive Input Representation

## How to train
```
python main.py --cutoffs='5000 15000' --cuda --data='data/wikitext-2' --save='wikitext_ckpt' --batch_size=60 --bptt=50 --lr=30 --no_log --no_save
```
