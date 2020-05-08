python train.py \
  --base_dir '/mnt1/tacotron' \
  --name 'lj-de-midtune' \
  --transfer_dir '/mnt1/tacotron-de' \
  --transfer_run 'de-fromscratch' \
  --restore_step 22000 # restore 40k step checkpoint from german
