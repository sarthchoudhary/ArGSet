#!/bin/bash
#SBATCH --job-name=initial_processing_ArgSet    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sarthak@camk.edu.pl   # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=08:00:00               # Time limit hrs:min:sec (you may not want this)
#SBATCH --output=%N-%j.out   # Standard output and error log
#SBATCH --mem=1024M                    # Allocated 250 megabytes of memory for the job.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:turing:1                # specify gpu


python read_analyze_ArgSet_data.py Run_0 [0,1,2] 2 0 30