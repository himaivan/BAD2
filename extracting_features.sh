#! /bin/bash

stage=0

if [ $stage -le 0 ];then

list_file=. # set-up directory

mkdir data

file_tr1=$list_file/BirdVox_label.csv
file_tr2=$list_file/ff1010bird_metadata.csv
file_tr3=$list_file/warblrb10k_public_metadata.csv

file_ls1=$list_file/poland.list.csv
file_ls2=$list_file/chern.list.csv
file_ls3=$list_file/wabrlrb10k.list.csv

echo $file_tr1
echo $file_tr2
echo $file_tr3

echo $file_ls1
echo $file_ls2
echo $file_ls3

matlab -r "featextract('$file_tr1')" -nojvm -nodisplay
matlab -r "featextract('$file_tr2')" -nojvm -nodisplay
matlab -r "featextract('$file_tr3')" -nojvm -nodisplay

matlab -r "featextract('$file_ls1')" -nojvm -nodisplay
matlab -r "featextract('$file_ls2')" -nojvm -nodisplay
matlab -r "featextract('$file_ls3')" -nojvm -nodisplay

mv *.mat data

fi

