#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>

#define KILL_PROCESS 2
#define KILL_ITER 3

#define m_printf if (myrank==0)printf
#define TMAX 20
#define NX 20
#define NY 30

int tmax = TMAX;
int nx = NX;
int ny = NY;
double start, end;

// flag - error in process
int error_flag = 0;
// killed one process
int flag_killed = 0;

// files with saved version of array
char file_xyz[20];

// communicator for working processes
MPI_Comm work_comm;
int myrank, ranksize;

static void init_array(int tmax, int nx, int ny, double ex[nx][ny], double ey[nx][ny], double hz[nx][ny])
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
            if (j == 0) fprintf(stderr, "\n");
            fprintf(stderr, "%0.2lf ", ey[i][j]);
        }
    }
    fprintf(stderr, "\nend   dump: %s\n", "ey");

    fprintf(stderr, "begin dump: %s", "HZ");
    for (i = 0; i < nx; i++)
    {
        for (j = 0; j < ny; j++)
        {
            if (j == 0) fprintf(stderr, "\n");
            fprintf(stderr, "%0.2lf ", hz[i][j]);
        }
    }
    fprintf(stderr, "\nend   dump: %s\n", "hz");
}

// Save arrays versions to files
static void save_checkpoint(int n_x, double ex[n_x][ny], double ey[n_x][ny], double hz[n_x][ny], char* file_xyz)
{
    
    FILE* file = fopen(file_xyz, "w");

    // Write to ex
    for (int i = 0; i < n_x; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            fprintf(file, "%0.20f ", ex[i][j]);
        }
        fprintf(file, "\n");
    }

    // Write to ey
    for (int i = 0; i < n_x; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            fprintf(file, "%0.20f ", ey[i][j]);
        }
        fprintf(file, "\n");
    }

    // Write to hz
    for (int i = 0; i < n_x; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            fprintf(file, "%0.20f ", hz[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

static void read_from_checkpoint(int n_x, double ex[n_x][ny], double ey[n_x][ny], double hz[n_x][ny], char* file_xyz)
{
    FILE* file = fopen(file_xyz, "r");

    // Read ex
    for (int i = 0; i < n_x; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            fscanf(file, "%lf ", &ex[i][j]);
        }
    }

    // Read ey
    for (int i = 0; i < n_x; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            fscanf(file, "%lf ", &ey[i][j]);
        }
    }

    // Read hz
    for (int i = 0; i < n_x; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            fscanf(file, "%lf ", &hz[i][j]);
        }
    }
    fclose(file);
}


static void err_handler(MPI_Comm* pcomm, int* perr, ...)
{
    int err = *perr;
    int size, nf, len;
    char errstr[MPI_MAX_ERROR_STRING];
    error_flag = 1;
    flag_killed = 1;
    MPI_Group group_f;

    MPI_Comm_size(work_comm, &size);
    MPIX_Comm_failure_ack(work_comm);
    MPIX_Comm_failure_get_acked(work_comm, &group_f);
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(err, errstr, &len);
    printf("\nRank %d / %d: Notified of error %s. %d found dead\n", myrank, size, errstr, nf);

    // new communicator for working processes
    MPIX_Comm_shrink(work_comm, &work_comm);
    MPI_Comm_rank(work_comm, &myrank);
    // change filename for file_xyz
    snprintf(file_xyz, 20, "check_point_%d.txt", myrank);
}

static void kernel_fdtd_2d(int tmax, int nx, int ny, int n_x, double ex[n_x][ny], double ey[n_x][ny], double hz[n_x][ny], int work_ranksize)
{
    int t, i, j;

    double* ey_next = (double*)malloc(ny * sizeof(double));
    double* hz_prev = (double*)malloc(ny * sizeof(double));

    MPI_Request req[2];
    MPI_Status stat[2];

    for (t = 0; t < tmax; t++)
    {
        if (myrank == KILL_PROCESS && t == KILL_ITER && !flag_killed)
        {
            // kill one process
            raise(SIGKILL);
        }

        check_point:
        MPI_Barrier(work_comm);
        if (myrank < work_ranksize)
        {
            read_from_checkpoint(n_x, ex, ey, hz, file_xyz);

            if (myrank == 0)
            {
                // fill first row of ey
                for (j = 0; j < ny; j++)
                    ey[0][j] = (double)t;

                // send last row to next process
                MPI_Isend(&hz[n_x - 1][0], ny, MPI_DOUBLE, myrank + 1, 1, work_comm, req);

                if (error_flag == 1)
                {
                    error_flag = 0;
                    goto check_point;
                }

                for (i = 1; i < n_x; ++i)
                {
                    for (j = 0; j < ny; ++j)
                        ey[i][j] -= 0.5f * (hz[i][j] - hz[i - 1][j]);
                }

                MPI_Waitall(1, req, stat);
            }
            else if (myrank == work_ranksize - 1)
            {
                // receive last row from prev process
                MPI_Irecv(hz_prev, ny, MPI_DOUBLE, myrank - 1, 1, work_comm, req);

                if (error_flag == 1)
                {
                    error_flag = 0;
                    goto check_point;
                }
                MPI_Waitall(1, req, stat);

            }
            // rank != 0 and rank != work_ranksize - 1
            else
            {
                // receive last row from prev process
                MPI_Irecv(hz_prev, ny, MPI_DOUBLE, myrank - 1, 1, work_comm, req);
                // send last row to next process
                MPI_Isend(&hz[n_x - 1][0], ny, MPI_DOUBLE, myrank + 1, 1, work_comm, req + 1);

                if (error_flag == 1)
                {
                    error_flag = 0;
                    goto check_point;
                }
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
                MPI_Irecv(ey_next, ny, MPI_DOUBLE, myrank + 1, 1, work_comm, req);

                if (error_flag == 1)
                {
                    error_flag = 0;
                    goto check_point;
                }
                MPI_Waitall(1, req, stat);
            }
            else if (myrank == work_ranksize - 1)
            {
                // send first row to prev process
                MPI_Isend(&ey[0][0], ny, MPI_DOUBLE, myrank - 1, 1, work_comm, req);

                if (error_flag == 1)
                {
                    error_flag = 0;
                    goto check_point;
                }
                for (i = 0; i < n_x - 1; i++)
                {
                    for (j = 0; j < ny - 1; j++)
                        hz[i][j] -= 0.7 * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
                }
                MPI_Waitall(1, req, stat);
            }
            else
            {
                // receive first row from next process
                MPI_Irecv(ey_next, ny, MPI_DOUBLE, myrank + 1, 1, work_comm, req);
                // send first row to prev process
                MPI_Isend(&ey[0][0], ny, MPI_DOUBLE, myrank - 1, 1, work_comm, req + 1);

                if (error_flag == 1)
                {
                    error_flag = 0;
                    goto check_point;
                }
                MPI_Waitall(2, req, stat);
            }

            // for processes != ranksize
            if (myrank != work_ranksize - 1)
            {
                if (n_x == 1)
                {
                    for (j = 0; j < ny - 1; j++)
                        hz[0][j] -= 0.7 * (ex[0][j + 1] - ex[0][j] + ey_next[j] - ey[0][j]);
                }
                else
                {
                    for (i = 0; i < n_x; ++i)
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
        if (error_flag == 1)
        {
            error_flag = 0;
            goto check_point;
        }
        MPI_Barrier(work_comm);
        if (myrank < work_ranksize)
        {
            save_checkpoint(n_x, ex, ey, hz, file_xyz);
        }

    }

    free((void*)ey_next);
    free((void*)hz_prev);

}


int main(int argc, char** argv)
{

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranksize);
    work_comm = MPI_COMM_WORLD;

    // initial number of working processes
    int work_ranksize = ranksize - 1;
    // reserve process for errors
    int reserv_proc = ranksize - 1;

    // create error handler
    MPI_Errhandler err_h;
    MPI_Comm_create_errhandler(err_handler, &err_h);
    MPI_Comm_set_errhandler(work_comm, err_h);

    MPI_Barrier(work_comm);

    // create file names for process
    snprintf(file_xyz, 20, "check_point_%d.txt", myrank);
    printf("File name for proc %d - %s\n", myrank, file_xyz);

    if (nx % work_ranksize != 0)
    {
        fprintf(stderr, "The number of rows in the array must be completely divisible by the chosen number of processes, not counting one reserve process.\n"
            "Current number of rows = %d, chosen number of processes = %d.\n", nx, ranksize);
        MPI_Finalize();
        return 0;
    }

    // arrays for result
    double(*ex)[nx][ny]; ex = (double(*)[nx][ny])malloc((nx) * (ny) * sizeof(double));
    double(*ey)[nx][ny]; ey = (double(*)[nx][ny])malloc((nx) * (ny) * sizeof(double));
    double(*hz)[nx][ny]; hz = (double(*)[nx][ny])malloc((nx) * (ny) * sizeof(double));

    int n_x = (nx / work_ranksize);
    // printf(" In process %d rows = %d\n", myrank, n_x);
    // arrays for local work
    double(*ex_local)[n_x][ny]; ex_local = (double(*)[n_x][ny])malloc(n_x * ny * sizeof(double));
    double(*ey_local)[n_x][ny]; ey_local = (double(*)[n_x][ny])malloc(n_x * ny * sizeof(double));
    double(*hz_local)[n_x][ny]; hz_local = (double(*)[n_x][ny])malloc(n_x * ny * sizeof(double));

    if (myrank == 0)
    {
        // fill arrays
        init_array(tmax, nx, ny, *ex, *ey, *hz);
        // fprintf(stderr, "MAIN ARRAYS\n");
        // print_array(nx, ny, *ex, *ey, *hz);

        start = MPI_Wtime();

        MPI_Request req[work_ranksize * 3];

        // bias for other processes
        int bias = 0;
        int proc_x;
        for (int i = 0; i < work_ranksize; ++i)
        {
            proc_x = (nx / work_ranksize);
            // send part of main arrays to processes
            MPI_Isend((double*)ex + bias, (proc_x * ny), MPI_DOUBLE, i, 1, work_comm, (req + 3 * i));
            MPI_Isend((double*)ey + bias, (proc_x * ny), MPI_DOUBLE, i, 2, work_comm, (req + 3 * i + 1));
            MPI_Isend((double*)hz + bias, (proc_x * ny), MPI_DOUBLE, i, 3, work_comm, (req + 3 * i + 2));
            bias += (nx / work_ranksize) * ny;
        }
    }

    // fill local arrays
    if (myrank < reserv_proc)
    {
        // printf(" Process %d size = %d \n", myrank, n_x * ny);
        MPI_Recv(ex_local, (n_x * ny), MPI_DOUBLE, 0, 1, work_comm, MPI_STATUS_IGNORE);
        MPI_Recv(ey_local, (n_x * ny), MPI_DOUBLE, 0, 2, work_comm, MPI_STATUS_IGNORE);
        MPI_Recv(hz_local, (n_x * ny), MPI_DOUBLE, 0, 3, work_comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(work_comm);
    // save arrays version
    if (myrank < reserv_proc)
    {
        save_checkpoint(n_x, *ex_local, *ey_local, *hz_local, file_xyz);
    }

    kernel_fdtd_2d(tmax, nx, ny, n_x, *ex_local, *ey_local, *hz_local, work_ranksize);

    MPI_Request req[(work_ranksize + 1) * 3];
    MPI_Status stat[(work_ranksize + 1) * 3];

    if (myrank == 0)
    {
        // bias for other processes
        int bias = 0;
        int proc_x;
        for (int i = 0; i < work_ranksize; ++i)
        {
            proc_x = (nx / work_ranksize);
            // receive part of main arrays to processes
            MPI_Irecv((double*)ex + bias, (proc_x * ny), MPI_DOUBLE, i, 1, work_comm, (req + 3 * i + 3));
            MPI_Irecv((double*)ey + bias, (proc_x * ny), MPI_DOUBLE, i, 2, work_comm, (req + 3 * i + 4));
            MPI_Irecv((double*)hz + bias, (proc_x * ny), MPI_DOUBLE, i, 3, work_comm, (req + 3 * i + 5));
            bias += (nx / work_ranksize) * ny;
        }

    }

    if (myrank < reserv_proc)
    {
        // send local arrays
        // printf(" Process %d size = %d \n", myrank, n_x * ny);
        MPI_Isend(ex_local, (n_x * ny), MPI_DOUBLE, 0, 1, work_comm, req);
        MPI_Isend(ey_local, (n_x * ny), MPI_DOUBLE, 0, 2, work_comm, req + 1);
        MPI_Isend(hz_local, (n_x * ny), MPI_DOUBLE, 0, 3, work_comm, req + 2);
    }

    if (myrank == 0)
    {
        MPI_Waitall((work_ranksize + 1) * 3, req, stat);
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
    }

    free((void*)ex);
    free((void*)ey);
    free((void*)hz);

    free((void*)ex_local);
    free((void*)ey_local);
    free((void*)hz_local);

    remove(file_xyz);

    MPI_Finalize();
    return 0;
}
