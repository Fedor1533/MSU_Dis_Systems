#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define m_printf if (myrank==0)printf
#define TMAX 20
#define NX 20
#define NY 30

int tmax = TMAX;
int nx = NX;
int ny = NY;
double start, end;

static void init_array (int tmax, int nx, int ny, double ex[nx][ny], double ey[nx][ny], double hz[nx][ny])
{
  int i, j;
  for (i = 0; i < nx; i++)
  {
      for (j = 0; j < ny; j++)
      {
          ex[i][j] = ((double)i * (j + 1)) / nx;
          ey[i][j] = ((double)i * (j + 2)) / ny;
          hz[i][j] = ((double)i * (j + 3)) / nx;
      }
  }
  
}

static void print_array(int nx, int ny, double ex[nx][ny], double ey[nx][ny], double hz[nx][ny])
{
  int i, j;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "EX");
  for (i = 0; i < nx; i++)
  {
      for (j = 0; j < ny; j++) 
      {
          if (j == 0) fprintf(stderr, "\n");
          fprintf(stderr, "%0.2lf ", ex[i][j]);
      }
  }
  fprintf(stderr, "\nend   dump: %s\n", "ex");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");

  fprintf(stderr, "begin dump: %s", "EY");
  for (i = 0; i < nx; i++)
  {
      for (j = 0; j < ny; j++) 
      {
          if (j== 0) fprintf(stderr, "\n");
          fprintf(stderr, "%0.2lf ", ey[i][j]);
      }
  }
  fprintf(stderr, "\nend   dump: %s\n", "ey");

  fprintf(stderr, "begin dump: %s", "HZ");
  for (i = 0; i < nx; i++)
  {
      for (j = 0; j < ny; j++) 
      {
          if (j== 0) fprintf(stderr, "\n");
          fprintf(stderr, "%0.2lf ", hz[i][j]);
      }
  }
  fprintf(stderr, "\nend   dump: %s\n", "hz");
}

static void kernel_fdtd_2d(int tmax, int nx, int ny, int n_x, double ex[n_x][ny], double ey[n_x][ny], double hz[n_x][ny], int myrank, int ranksize)
{
  int t, i, j;
  printf(" Kernel process %d rows = %d\n", myrank,  n_x);
  
  double *ey_next= (double *) malloc(ny * sizeof(double));
  double *hz_prev = (double *) malloc(ny * sizeof(double));
  
  MPI_Request req[2];
  MPI_Status stat[2];
  
  for(t = 0; t < tmax; t++)
  {
    if (myrank == 0)
    {
    	// fill first row of ey
        for (j = 0; j < ny; j++)
            ey[0][j] = (double) t;
        
    	// send last row to next process
    	MPI_Isend(&hz[n_x - 1][0], ny, MPI_DOUBLE, myrank + 1, 1, MPI_COMM_WORLD, req);
    
      	for (i = 1; i < n_x; ++i) 
      	{
      	  for (j = 0; j < ny; ++j) 
             ey[i][j] -= 0.5f * (hz[i][j] - hz[i - 1][j]);
        }
        
      	MPI_Waitall(1, req, stat);
    }
    else if (myrank == ranksize - 1)
    {
      	// receive last row from prev process
      	MPI_Irecv(hz_prev, ny, MPI_DOUBLE, myrank - 1, 1, MPI_COMM_WORLD, req);
      	MPI_Waitall(1, req, stat);
      	 
    }
    // rank != 0 and rank != ranksize - 1
    else
    {
      	// receive last row from prev process
      	MPI_Irecv(hz_prev, ny, MPI_DOUBLE, myrank - 1, 1, MPI_COMM_WORLD, req);
      	// send last row to next process
      	MPI_Isend(&hz[n_x - 1][0], ny, MPI_DOUBLE, myrank + 1, 1, MPI_COMM_WORLD, req+1);
      	MPI_Waitall(2, req, stat);	
    }
    
    if (myrank != 0)
    {
    	for (i = 0; i < n_x; ++i) 
    	{
	   for (j = 0; j < ny; ++j) 
	   {
	      if (i == 0) 
	      {
		 ey[i][j] -= 0.5f * (hz[i][j] - hz_prev[j]);
	      } 
	      else 
	      {
		 ey[i][j] -= 0.5f * (hz[i][j] - hz[i - 1][j]);
	      }
	   }
    	}
    }
    
    // same for all processes
    for (i = 0; i < n_x; ++i) 
    {
      for (j = 1; j < ny; ++j)
        ex[i][j] -= 0.5f * (hz[i][j] - hz[i][j - 1]);
    }	
    
    if (myrank == 0) 
    {
      // receive first row from next process
      MPI_Irecv(ey_next, ny, MPI_DOUBLE, myrank + 1, 1, MPI_COMM_WORLD, req);
      MPI_Waitall(1, req, stat);
    } 
    else if (myrank == ranksize - 1)
    {
      // send first row to prev process
      MPI_Isend(&ey[0][0], ny, MPI_DOUBLE, myrank - 1, 1, MPI_COMM_WORLD, req);
      for (i = 0; i < n_x - 1; i++)
      {
        for (j = 0; j < ny - 1; j++)
            hz[i][j] -= 0.7* (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
      }
      MPI_Waitall(1, req, stat);
    }
    else
    {
      // receive first row from next process
      MPI_Irecv(ey_next, ny, MPI_DOUBLE, myrank + 1, 1, MPI_COMM_WORLD, req);
      // send first row to prev process
      MPI_Isend(&ey[0][0], ny, MPI_DOUBLE, myrank - 1, 1, MPI_COMM_WORLD, req+1);
      MPI_Waitall(2, req, stat);
    }
    
    // for processes != ranksize
    if (myrank != ranksize - 1)
    {
    	if (n_x == 1)
    	{
    	  for (j = 0; j < ny - 1; j++)
            hz[0][j] -= 0.7* (ex[0][j+1] - ex[0][j] + ey_next[j] - ey[0][j]);
    	}
    	else
    	{
    	  for(i = 0; i < n_x; ++i) 
    	  {
            for (j = 0; j < ny - 1; ++j) 
            {
          	if (i + 1 >= n_x) 
          	{
            	   hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey_next[j] - ey[i][j]);
          	} 
          	else 
          	{
            	   hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
          	}	
            }
      	  }	
    	}	
    }
     
  }
  
  free((void *) ey_next);
  free((void *) hz_prev);
  	
}


int main(int argc, char** argv)
{
	
  int myrank, ranksize;	
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
  MPI_Barrier(MPI_COMM_WORLD);
  
  double (*ex)[nx][ny];
  double (*ey)[nx][ny];
  double (*hz)[nx][ny];
  
  int n_x = (nx / ranksize + (nx % ranksize > myrank));
  printf(" In process %d rows = %d\n", myrank,  n_x);
  double (*ex_local)[n_x][ny]; ex_local = (double(*)[n_x][ny])malloc(n_x * ny * sizeof(double));
  double (*ey_local)[n_x][ny]; ey_local = (double(*)[n_x][ny])malloc(n_x * ny * sizeof(double));
  double (*hz_local)[n_x][ny]; hz_local = (double(*)[n_x][ny])malloc(n_x * ny * sizeof(double));
  
  if(myrank == 0)
  {
      // create arrays
      ex = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));
      ey = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));
      hz = (double(*)[nx][ny])malloc ((nx) * (ny) * sizeof(double));
      
      // fill arrays
      init_array (tmax, nx, ny, *ex, *ey, *hz);
      fprintf(stderr, "MAIN ARRAYS\n");
      print_array(nx, ny, *ex, *ey, *hz);
	  
      start = MPI_Wtime();
	  
      MPI_Request req[ranksize*3];
	  
      // bias for other processes
      int bias = 0;
	  int proc_x;
      for (int i = 0; i < ranksize; ++i)
      {
          proc_x = (nx / ranksize + (nx % ranksize > i));
	  // printf(" For process %d rows = %d bias = %d \n", i, proc_x, bias);
          // send part of main arrays to processes
          MPI_Isend((double *)ex + bias, (proc_x * ny), MPI_DOUBLE, i, 1, MPI_COMM_WORLD, (req+3*i));
          MPI_Isend((double *)ey + bias, (proc_x * ny), MPI_DOUBLE, i, 2, MPI_COMM_WORLD, (req+3*i+1));
          MPI_Isend((double *)hz + bias, (proc_x * ny), MPI_DOUBLE, i, 3, MPI_COMM_WORLD, (req+3*i+2));
          bias += (nx / ranksize + (nx % ranksize > i)) * ny;
      }
  }
  
  // fill local arrays
  printf(" Process %d size = %d \n", myrank, n_x*ny);
  MPI_Recv(ex_local, (n_x * ny), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(ey_local, (n_x * ny), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(hz_local, (n_x * ny), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
  MPI_Barrier(MPI_COMM_WORLD);

  kernel_fdtd_2d (tmax, nx, ny, n_x, *ex_local, *ey_local, *hz_local, myrank, ranksize);
  
  MPI_Request req[(ranksize+1)*3];
  MPI_Status stat[(ranksize+1)*3];
  
  if(myrank == 0)
  {
      // bias for other processes
      int bias = 0;
	  int proc_x;
      for (int i = 0; i < ranksize; ++i)
      {
      	  proc_x = (nx / ranksize + (nx % ranksize > i));
          // receive part of main arrays to processes
          MPI_Irecv((double *)ex + bias, (proc_x * ny), MPI_DOUBLE, i, 1, MPI_COMM_WORLD, (req+3*i+3));
          MPI_Irecv((double *)ey + bias, (proc_x * ny), MPI_DOUBLE, i, 2, MPI_COMM_WORLD, (req+3*i+4));
          MPI_Irecv((double *)hz + bias, (proc_x * ny), MPI_DOUBLE, i, 3, MPI_COMM_WORLD, (req+3*i+5));
          bias += (nx / ranksize + (nx % ranksize > i)) * ny;
      }
	  
  }
  
  // send local arrays
  printf(" Process %d size = %d \n", myrank, n_x*ny);
  MPI_Isend(ex_local, (n_x * ny), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, req);
  MPI_Isend(ey_local, (n_x * ny), MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, req+1);
  MPI_Isend(hz_local, (n_x * ny), MPI_DOUBLE, 0, 3, MPI_COMM_WORLD, req+2);
  
  if (myrank == 0)
  {
	  MPI_Waitall((ranksize+1)*3, req, stat);
  }
  else
  {
	 MPI_Waitall(3, req, stat); 
  }
  
  if (myrank == 0)
  {
     fprintf(stderr, "NEW MAIN ARRAYS\n");
     print_array(nx, ny, *ex, *ey, *hz);
     
     end = MPI_Wtime();
     printf("Time of task = %f\n", end - start);
     
     free((void*)ex);
     free((void*)ey);
     free((void*)hz);
  }	  
  
  free((void*)ex_local);
  free((void*)ey_local);
  free((void*)hz_local);
  
  MPI_Finalize();
  return 0;
}

