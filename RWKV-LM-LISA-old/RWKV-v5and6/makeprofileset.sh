#!/bin/bash

N_LAYER="24"
H_LAYER=$(($N_LAYER / 2))

DestDir="layerprofile"

DestName="${DestDir}/${N_LAYER}_Increase.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 1.2 --endvalue 1.4 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_Decrease.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1.4 --centroidvalue 1.2 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_Mountain.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 1.4 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_Valley.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 0.6 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_IncreaseExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 0 --centroidvalue 0.5 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_DecreaseExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 0.5 --endvalue 0 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_MountainExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 0 --centroidvalue 1 --endvalue 0 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_ValleyExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 0  --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_FlatExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 1 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName


N_LAYER="32"

DestName="${DestDir}/${N_LAYER}_Increase.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 1.2 --endvalue 1.4 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_Decrease.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1.4 --centroidvalue 1.2 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_Mountain.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 1.4 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_Valley.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 0.6 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_IncreaseExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 0 --centroidvalue 0.5 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_DecreaseExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 0.5 --endvalue 0 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_MountainExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 0 --centroidvalue 1 --endvalue 0 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_ValleyExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 0  --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName

DestName="${DestDir}/${N_LAYER}_FlatExtreme.csv"
python makeprofile.py \
 --num_layer $N_LAYER --startvalue 1 --centroidvalue 1 --endvalue 1 --centroidpos $H_LAYER \
 --outputfile $DestName