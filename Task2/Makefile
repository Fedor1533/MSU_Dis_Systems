fdtd-2d_MPI_FT: fdtd-2d_MPI_FT.out
	mpirun -np 5 --map-by :OVERSUBSCRIBE --with-ft ulfm ./fdtd-2d_MPI_FT.out
fdtd-2d_MPI_FT.out: fdtd-2d_MPI_FT.c
	mpicc fdtd-2d_MPI_FT.c -o fdtd-2d_MPI_FT.out
clean:
	rm -rf fdtd-2d_MPI_FT.out
