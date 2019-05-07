#!/bin/bash
#
##SBATCH --nodes=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --job-name=JacobiWeakScale
#SBATCH --mail-type=END
##SBATCH --mail-user=aem578@nyu@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/aem578/homework2submission/hw6

module purge
module load openmpi/intel/3.1.3

make

mpirun -np 16 ./jacobi_mpi 1600 100
mpirun -np 16 ./jacobi_mpi 1600 1000
mpirun -np 16 ./jacobi_mpi 1600 10000