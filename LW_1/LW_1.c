#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "../header.h"

#define N (5000)
#define DEL_X (0.5)
#define DEL_Y (0.25)

void start(int, int, size_t, dbl, size_t, dbl, int, const char * restrict);

static void init(int, int, size_t, size_t,
    dbl (* restrict * restrict)[],
    dbl (* restrict * restrict)[],
    dbl (* restrict * restrict)[]);
static void finalize(size_t, dbl (* restrict)[], dbl (* restrict)[],
    dbl (* restrict)[]);
static void send_recv(int, int, size_t, size_t, dbl (* restrict)[]);
static void send(int, int, size_t, size_t, dbl (* restrict)[]);
static void recv(int, int, size_t, size_t, dbl (* restrict)[]);
static void file_write(int, int, size_t, size_t, dbl (* restrict)[],
    const char * restrict);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <m rows> <m columns> <file out>\n", argv[0]);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_LASTCODE);
    }

    const ptrdiff_t m_x = strtoll(argv[1], NULL, 10) + 2,
                    m_y = strtoll(argv[2], NULL, 10) + 2;
    mpi_abort_if(m_x < 0 || m_y < 0, "m_x < 0 || m_y < 0");

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    start(size, rank, m_x, DEL_X, m_y, DEL_Y, N, argv[3]);

    MPI_Finalize();

    return 0;
}

void start(const int size, const int rank,
    const size_t m_x, const dbl del_x, const size_t m_y, const dbl del_y,
    const int n, const char * const restrict filename)
{
    mpi_abort_if(m_x < 3U || del_x < DBL_EPSILON || isinf(del_x) || isnan(del_x)
        || m_y < 3U || del_y < DBL_EPSILON || isinf(del_y) || isnan(del_y)
        || size < 1 || rank < 0 || size <= rank || m_x - 2U < (size_t) size,
        "m_x, del_x, m_y, del_y, size, rank");

    dbl (* restrict u_k)[m_y],
        (* restrict u_k_plus_1)[m_y],
        (* restrict r)[m_y];
    const size_t rows = (m_x - 2U) / size + ((size_t) rank < (m_x - 2U) % size) + 2U;
    init(size, rank, rows, m_y, &u_k, &u_k_plus_1, &r);
    const dbl rdx2 = 1.0 / del_x / del_x,
              rdy2 = 1.0 / del_y / del_y,
              beta = 1.0 / (2.0 * (rdx2 + rdy2));
    const size_t rows_minus_1 = rows - 1,
                 m_y_minus_1 = m_y - 1;
    dbl wtime_start = MPI_Wtime(), wtime_end;
    for (int k = 0; k < n; ++k)
    {
        send_recv(size, rank, rows, m_y, u_k);
        for (size_t i = 1; i < rows_minus_1; ++i)
        {
            for (size_t j = 1; j < m_y_minus_1; ++j)
            {
                u_k_plus_1[i][j] = ((u_k[i - 1][j] + u_k[i + 1][j]) * rdx2 +
                    (u_k[i][j - 1] + u_k[i][j + 1]) * rdy2 - r[i][j]) * beta;
            }
        }
        void *tmp_ptr = (void *) u_k;
        u_k = u_k_plus_1;
        u_k_plus_1 = (dbl (*)[m_y]) tmp_ptr;
    }
    wtime_end = MPI_Wtime();
    printf("wtime = %lf\n", wtime_end - wtime_start);

    file_write(size, rank, rows, m_y, u_k, filename);
    finalize(m_y, u_k, u_k_plus_1, r);
}

static void init(const int size, const int rank,
    const size_t rows, const size_t m_y,
    dbl (* restrict * const restrict u_k)[m_y],
    dbl (* restrict * const restrict u_k_plus_1)[m_y],
    dbl (* restrict * const restrict r)[m_y])
{
    *u_k        = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    *u_k_plus_1 = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    *r          = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    mpi_abort_if(!*u_k || !*u_k_plus_1 || !*r, "malloc()");

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < m_y; ++j)
        {
            if ((!rank && !i) || (rank + 1 == size && i + 1 == rows)
                || !j || j + 1 == m_y)
            {
                (*u_k_plus_1)[i][j] = (*u_k)[i][j] = 1.0;
            }
            else
            {
                (*u_k_plus_1)[i][j] = (*u_k)[i][j] = 0.0;
            }
            (*r)[i][j] = 0.0;
        }
    }
}

static void finalize(const size_t m_y, dbl (* const restrict u_k)[m_y],
    dbl (* const restrict u_k_plus_1)[m_y],
    dbl (* const restrict r)[m_y])
{
    free(u_k);
    free(u_k_plus_1);
    free(r);
}

static void send_recv(const int size, const int rank,
    const size_t rows, const size_t m_y, dbl (* const restrict u_k)[m_y])
{
    if (rank % 2)
    {
        send(size, rank, rows, m_y, u_k);
        recv(size, rank, rows, m_y, u_k);
    }
    else
    {
        recv(size, rank, rows, m_y, u_k);
        send(size, rank, rows, m_y, u_k);
    }
}

static void send(const int size, const int rank,
    const size_t rows, const size_t m_y, dbl (* const restrict u_k)[m_y])
{
    int int_return;
    if (rank)
    {
        int_return = MPI_Send(u_k[1] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank - 1, 0, MPI_COMM_WORLD);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Send()");
    }
    if (rank + 1 < size)
    {
        int_return = MPI_Send(u_k[rows - 2U] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Send()");
    }
}

static void recv(const int size, const int rank,
    const size_t rows, const size_t m_y, dbl (* const restrict u_k)[m_y])
{
    int int_return;
    MPI_Status status;
    if (rank + 1 < size)
    {
        int_return = MPI_Recv(u_k[rows - 1] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD, &status);
        mpi_abort_if(int_return != MPI_SUCCESS
            || status.MPI_ERROR != MPI_SUCCESS, "MPI_Recv()");
    }
    if (rank)
    {
        int_return = MPI_Recv(u_k[0] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank - 1, 0, MPI_COMM_WORLD, &status);
        mpi_abort_if(int_return != MPI_SUCCESS
            || status.MPI_ERROR != MPI_SUCCESS, "MPI_Recv()");
    }
}

static void file_write(const int size, const int rank,
    const size_t rows, const size_t m_y, dbl (* const restrict u_k)[m_y],
    const char * const restrict filename)
{
    int int_return;
    FILE * restrict file_out;
    if (!rank)
    {
        file_out = fopen(filename, "wb");
        mpi_abort_if(!file_out, "fopen()");
        int_return = fclose(file_out);
        mpi_abort_if(int_return == EOF, "fclose()");
    }

    int_return = MPI_Barrier(MPI_COMM_WORLD);
    mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Barrier()");

    const size_t rows_minus_1 = rows - 1, m_y_minus_2 = m_y - 2U;
    for (int r = 0; r < size; ++r)
    {
        if (r == rank)
        {
            file_out = fopen(filename, "ab");
            mpi_abort_if(!file_out, "fopen()");
            for (size_t i = 1; i < rows_minus_1; ++i)
            {
                size_t size_return = fwrite(u_k[i] + 1, sizeof(dbl), m_y_minus_2, file_out);
                mpi_abort_if(size_return != m_y_minus_2, "fwrite()");
            }
            int_return = fclose(file_out);
            mpi_abort_if(int_return == EOF, "fclose()");
        }
        int_return = MPI_Barrier(MPI_COMM_WORLD);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Barrier()");
    }
}
