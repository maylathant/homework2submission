#!/bin/bash
#
##SBATCH --nodes=16
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=sample_sort_time
#SBATCH --mail-type=END
##SBATCH --mail-user=aem578@nyu@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/aem578/homework2submission/hw6

module purge
module load openmpi/intel/3.1.3

make


mpirun -np 4 ./sample_sort 10000
mpirun -np 4 ./sample_sort 100000
mpirun -np 4 ./sample_sort 1000000
mpirun -np 4 ./sample_sort 10000000

mpirun -np 10 ./sample_sort 10000
mpirun -np 10 ./sample_sort 100000
mpirun -np 10 ./sample_sort 1000000
mpirun -np 10 ./sample_sort 10000000

mpirun -np 16 ./sample_sort 10000
mpirun -np 16 ./sample_sort 100000
mpirun -np 16 ./sample_sort 1000000
mpirun -np 16 ./sample_sort 10000000

mpirun -np 50 ./sample_sort 10000
mpirun -np 50 ./sample_sort 100000
mpirun -np 50 ./sample_sort 1000000
mpirun -np 50 ./sample_sort 10000000

mpirun -np 100 ./sample_sort 10000
mpirun -np 100 ./sample_sort 100000
mpirun -np 100 ./sample_sort 1000000
mpirun -np 100 ./sample_sort 10000000

mpirun -np 200 ./sample_sort 10000
mpirun -np 200 ./sample_sort 100000
mpirun -np 200 ./sample_sort 1000000
mpirun -np 200 ./sample_sort 10000000

mpirun -np 4 ./sample_sort 40000
mpirun -np 10 ./sample_sort 100000
mpirun -np 16 ./sample_sort 160000
mpirun -np 32 ./sample_sort 320000
mpirun -np 50 ./sample_sort 500000
mpirun -np 64 ./sample_sort 640000
mpirun -np 100 ./sample_sort 1000000
mpirun -np 200 ./sample_sort 2000000