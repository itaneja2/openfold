#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --output=get_conformational_states.out
#SBATCH --error=get_conformational_states.err

python -u get_conformational_states.py > get_conformational_states.out
