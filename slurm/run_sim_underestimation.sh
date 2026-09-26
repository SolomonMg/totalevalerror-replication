#!/bin/bash
#SBATCH --job-name=tee_underest
#SBATCH --output=slurm/logs/underest_%j.out
#SBATCH --error=slurm/logs/underest_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_309_general

module purge
module load r/4.5.1

cd ~/total_llm_error

# Delete old CSVs to force re-run
rm -f data/processed/sim_underestimation.csv
rm -f data/processed/sim_underestimation_summary.csv
rm -f data/processed/sim_underestimation_20researchers.csv

Rscript analysis/04g_sim_underestimation.R --cores 16
