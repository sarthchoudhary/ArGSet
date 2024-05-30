#!/bin/bash
#SBATCH --job-name=initial_processing_ArgSet    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sarthak@camk.edu.pl   # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=01:30:00               # Time limit hrs:min:sec (you may not want this)
#SBATCH --output=%N-%j.out            # Standard output and error log
#SBATCH --mem=1024M                   # Allocated 250 megabytes of memory for the job.
#SBATCH --partition=short             # short < 2H; long, gpu, dgx etc.
###SBATCH --gres=gpu:turing:1                # specify gpu


python /home/sarthak/my_projects/argset/scripts/fingerplots_filter_TSpectrum_channel2.py
