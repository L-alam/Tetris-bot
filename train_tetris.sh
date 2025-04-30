#!/bin/bash
#$ -N tetris_training
#$ -l h_rt=8:00:00
#$ -l mem_per_core=8G
#$ -o tetris_output.log
#$ -e tetris_error.log
#$ -m ea
#$ -M your_email@bu.edu
cd /projectnb/cs440/students/lalam/tetris_project
mkdir -p params
java -cp "./lib/*:." edu.bu.pas.tetris.Main \
  -p 2000 \
  -t 5000 \
  -v 500 \
  -u 8 \
  -b 20000 \
  -m 512 \
  -n 1.0E-3 \
  -g 0.98 \
  -s \
  | tee training_log.log