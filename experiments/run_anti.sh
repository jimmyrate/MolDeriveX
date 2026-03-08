#!/bin/bash

for i in {128,42,7}
do
    python main_ren.py mol_selfies morbo $i 1000  /root/morbo/TransVAE-master/data/Gram-positive-for-mbo.csv  combined_antibiotic_momo_preds_abaucin.csv
    #echo $i
done