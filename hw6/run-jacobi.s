#!/bin/bash
#
##SBATCH --nodes=16
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
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

mpirun -np 256 ./jacobi_mpi 25600 100
mpirun -np 256 ./jacobi_mpi 25600 1000
mpirun -np 256 ./jacobi_mpi 25600 10000