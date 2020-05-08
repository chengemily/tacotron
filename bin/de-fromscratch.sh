#!/bin/bash
python train.py \
  --base_dir '/mnt1/tacotron-de' \
  --name 'de-fromscratch' \
  --restore_step 3000 \
  --hparams 'max_iters=310,cleaners=transliteration_cleaners,batch_size=32'
