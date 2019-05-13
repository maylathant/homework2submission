#!/bin/bash
#
##SBATCH --nodes=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --time=0:35:00
#SBATCH --mem=60GB
#SBATCH --job-name=Multigridaem
#SBATCH --mail-type=END
##SBATCH --mail-user=aem578@nyu@nyu.edu
#SBATCH --output=slurm_%j.out

cd /scratch/aem578/homework2submission/hw6

module purge
module load openmpi/intel/3.1.3

g++ -O3 -fopenmp multigrid_omp.cpp -o multigrid_omp

./multigrid_omp 700000000 1 4 1
./multigrid_omp 700000000 1 4 2
./multigrid_omp 700000000 1 4 3
./multigrid_omp 700000000 1 4 4
./multigrid_omp 700000000 1 4 5
./multigrid_omp 700000000 1 4 6
./multigrid_omp 700000000 1 4 7
./multigrid_omp 700000000 1 4 8
./multigrid_omp 700000000 1 4 9
./multigrid_omp 700000000 1 4 10
./multigrid_omp 700000000 1 4 11
./multigrid_omp 700000000 1 4 12
./multigrid_omp 700000000 1 4 13
./multigrid_omp 700000000 1 4 14
./multigrid_omp 700000000 1 4 15
./multigrid_omp 700000000 1 4 16
./multigrid_omp 700000000 1 4 17
./multigrid_omp 700000000 1 4 18
./multigrid_omp 700000000 1 4 19
./multigrid_omp 700000000 1 4 20

./multigrid_omp 100000000 1 4 1
./multigrid_omp 200000000 1 4 2
./multigrid_omp 300000000 1 4 3
./multigrid_omp 400000000 1 4 4
./multigrid_omp 500000000 1 4 5
./multigrid_omp 600000000 1 4 6
./multigrid_omp 700000000 1 4 7
./multigrid_omp 800000000 1 4 8
./multigrid_omp 900000000 1 4 9
./multigrid_omp 1000000000 1 4 10
./multigrid_omp 1100000000 1 4 11
./multigrid_omp 1200000000 1 4 12
./multigrid_omp 1300000000 1 4 13
./multigrid_omp 1400000000 1 4 14
./multigrid_omp 1500000000 1 4 15
./multigrid_omp 1600000000 1 4 16
./multigrid_omp 1700000000 1 4 17
./multigrid_omp 1800000000 1 4 18
./multigrid_omp 1900000000 1 4 19
./multigrid_omp 2000000000 1 4 20

