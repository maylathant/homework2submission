#!/bin/bash
#
##SBATCH --nodes=16
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=0:35:00
#SBATCH --mem=20GB
#SBATCH --job-name=JacobiStrongScale
#SBATCH --mail-type=END
##SBATCH --mail-user=aem578@nyu@nyu.edu
#SBATCH --output=slurm_strong%j.out

cd /scratch/aem578/homework2submission/hw6

module purge
module load openmpi/intel/3.1.3

make

mpirun -np 1 ./jacobi_mpi 25600 100
mpirun -np 4 ./jacobi_mpi 25600 100
mpirun -np 8 ./jacobi_mpi 25600 100
mpirun -np 16 ./jacobi_mpi 25600 100
mpirun -np 32 ./jacobi_mpi 25600 100
mpirun -np 64 ./jacobi_mpi 25600 100
mpirun -np 128 ./jacobi_mpi 25600 100
mpirun -np 256 ./jacobi_mpi 25600 100

