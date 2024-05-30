#!/bin/bash
#SBATCH --job-name=Advance_finger_plots_for_ArgSet    # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sarthak@camk.edu.pl   # Where to send mail
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=08:00:00               # Time limit hrs:min:sec (you may not want this)
#SBATCH --output=%N-%j.out            # Standard output and error log
#SBATCH --mem=256M                   # Allocated 250 megabytes of memory for the job.
#SBATCH --partition=short             # short < 2H; long, gpu, dgx etc.
###SBATCH --gres=gpu:turing:1                # specify gpu


python /home/sarthak/my_projects/argset/scripts/fingerplots_filter_TSpectrum_advance_channel2.py
