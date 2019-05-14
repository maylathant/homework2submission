#!/bin/bash
#
##SBATCH --nodes=16
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=0:35:00
#SBATCH --mem=20GB
#SBATCH --job-name=JacobiWeakScale
#SBATCH --mail-type=END
##SBATCH --mail-user=aem578@nyu@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/aem578/homework2submission/hw6

module purge
module load openmpi/intel/3.1.3

make

mpirun -np 1 ./jacobi_mpi 100 100
mpirun -np 1 ./jacobi_mpi 100 1000
mpirun -np 1 ./jacobi_mpi 100 10000

mpirun -np 4 ./jacobi_mpi 400 100
mpirun -np 4 ./jacobi_mpi 400 1000
mpirun -np 4 ./jacobi_mpi 400 10000


mpirun -np 16 ./jacobi_mpi 1600 100
mpirun -np 16 ./jacobi_mpi 1600 1000
mpirun -np 16 ./jacobi_mpi 1600 10000

mpirun -np 64 ./jacobi_mpi 6400 100
mpirun -np 64 ./jacobi_mpi 6400 1000
mpirun -np 64 ./jacobi_mpi 6400 10000

mpirun -np 256 ./jacobi_mpi 25600 100
mpirun -np 256 ./jacobi_mpi 25600 1000
mpirun -np 256 ./jacobi_mpi 25600 10000

mpirun -np 1 ./jacobi_noblock 100 100
mpirun -np 1 ./jacobi_noblock 100 1000
mpirun -np 1 ./jacobi_noblock 100 10000

mpirun -np 4 ./jacobi_noblock 400 100
mpirun -np 4 ./jacobi_noblock 400 1000
mpirun -np 4 ./jacobi_noblock 400 10000


mpirun -np 16 ./jacobi_noblock 1600 100
mpirun -np 16 ./jacobi_noblock 1600 1000
mpirun -np 16 ./jacobi_noblock 1600 10000

mpirun -np 64 ./jacobi_noblock 6400 100
mpirun -np 64 ./jacobi_noblock 6400 1000
mpirun -np 64 ./jacobi_noblock 6400 10000

mpirun -np 256 ./jacobi_noblock 25600 100
mpirun -np 256 ./jacobi_noblock 25600 1000
mpirun -np 256 ./jacobi_noblock 25600 10000