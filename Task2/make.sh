docker pull abouteiller/mpi-ft-ulfm
alias make='docker run -v $PWD:/sandbox:Z abouteiller/mpi-ft-ulfm make'
alias mpirun='docker run -v $PWD:/sandbox:Z abouteiller/mpi-ft-ulfm mpirun --oversubscribe -mca btl tcp,self'
make fdtd-2d_MPI_FT
