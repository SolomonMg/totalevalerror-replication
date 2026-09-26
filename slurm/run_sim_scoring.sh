#!/bin/bash
#SBATCH --job-name=tle_scoring_sim
#SBATCH --output=slurm/logs/scoring_sim_%j.out
#SBATCH --error=slurm/logs/scoring_sim_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_309_general

module purge
module load r/4.5.1

cd ~/total_llm_error
Rscript analysis/06_sim_scoring_recovery.R --nsim 500
