#include <errno.h>
#include <float.h>
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

int start(int, int, size_t, dbl, size_t, dbl, int, const char * restrict);
static int init(int, int, size_t, const size_t m_y,
    dbl (* restrict * restrict)[m_y],
    dbl (* restrict * restrict)[m_y],
    dbl (* restrict * restrict)[m_y]);
static int send_recv(int, int, size_t, size_t my, dbl (* restrict)[m_y]);
static int send(int, int, size_t, size_t my, dbl (* restrict)[m_y]);
static int recv(int, int, size_t, size_t my, dbl (* restrict)[m_y]);
static int file_write(int, int, size_t, size_t my, dbl (* restrict)[m_y],
    const char * restrict);

int main(int argc, const char *argv[])
{
    MPI_Init(&argc, &argv);

    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <m rows> <m columns> <file out>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const ptrdiff_t m_x = strtoll(argv[1], NULL, 10) + 2,
                    m_y = strtoll(argv[2], NULL, 10) + 2;
    exit_if(m_x < 0 || m_y < 0, "m_x < 0 || m_y < 0");

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int int_return = start(size, rank, m_x, DEL_X, m_y, DEL_Y, N, argv[3]);
    exit_if(int_return, err_0);

    MPI_Finalize();

    return 0;
}

int start(const int size, const int rank,
    const size_t m_x, const dbl del_x, const size_t m_y, const dbl del_y,
    const int n, const char * const restrict filename)
{
    dbl (* restrict u_k)[m_y],
        (* restrict u_k_plus_1)[m_y],
        (* restrict r)[m_y];

    int return_value = MPI_ERR_LASTCODE;
    err_if(m_x < 3U || del_x < DBL_EPSILON || isinf(del_x) || isnan(del_x)
        || m_y < 3U || del_y < DBL_EPSILON || isinf(del_y) || isnan(del_y)
        || size < 1 || rank < 0 || size <= rank || m_x - 2U < (size_t) size, err_0);

    const size_t rows = (m_x - 2U) / size + (rank < (m_x - 2U) % size) + 2U;
    const dbl rdx2 = 1.0 / del_x / del_x,
              rdy2 = 1.0 / del_y / del_y,
              beta = 1.0 / (2.0 * (rdx2 + rdy2));

    return_value = init(size, rank, rows, m_y, &u_k, &u_k_plus_1, &r);
    err_if(return_value != MPI_SUCCESS, err_0);

    const size_t rows_minus_1 = rows - 1,
                 m_y_minus_1 = m_y - 1;
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

    file_write();
    return_value = 0;

err_1:
    finalize();
err_0:
    return return_value;
}

static int init(const int size, const int rank, const size_t rows, const size_t m_y,
    dbl (* restrict * const restrict u_k)[m_y],
    dbl (* restrict * const restrict u_k_plus_1)[m_y],
    dbl (* restrict * const restrict r)[m_y])
{
    *u_k        = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    *u_k_plus_1 = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    *r          = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    err_if(!*u_k || !*u_k_plus_1 || !*r, err_0);

    for (ptrdiff_t i = 0; i < rows; ++i)
    {
        for (ptrdiff_t j = 0; j < m_y; ++j)
        {
            if ((!rank && !i) || (rank + 1 == size && i + 1 == rows)
                || !j || j + 1 == m_y)
            {
                (*u_k)[i][j] = 1.0;
            }
            else
            {
                (*u_k)[i][j] = 0.0;
            }
            (*r)[i][j] = 0.0;
        }
    }

    return MPI_SUCCESS;

err_0:
    free(*r);
    free(*u_k_plus_1);
    free(*u_k);

    return MPI_ERR_NO_MEM;
}

static int send_recv(const int size, const int rank,
    const size_t rows, const size_t m_y, dbl (* const restrict u_k)[m_y])
{
    int int_return = MPI_SUCCESS;
    if (rank % 2)
    {
        int_return = send(size, rank, rows, m_y, u_k);
        err_if(int_return != MPI_SUCCESS, err_0);
        int_return = recv(size, rank, rows, m_y, u_k);
        err_if(int_return != MPI_SUCCESS, err_0);
    }
    else
    {
        int_return = recv(size, rank, rows, m_y, u_k);
        err_if(int_return != MPI_SUCCESS, err_0);
        int_return = send(size, rank, rows, m_y, u_k);
        err_if(int_return != MPI_SUCCESS, err_0);
    }

err_0:
    return int_return;
}

static int send(const int size, const int rank,
    const size_t rows, const size_t m_y, dbl (* const restrict u_k)[m_y])
{
    int int_return = MPI_SUCCESS;
    if (rank)
    {
        int_return = MPI_Send(u_k[1] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank - 1, 0, MPI_COMM_WORLD);
        err_if(int_return != MPI_SUCCESS, err_0);
    }
    if (rank + 1 < size)
    {
        int_return = MPI_Send(u_k[rows - 2U] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD);
        err_if(int_return != MPI_SUCCESS, err_0);
    }

err_0:
    return int_return;
}

static int recv(const int size, const int rank,
    const size_t rows, const size_t m_y, dbl (* const restrict u_k)[m_y])
{
    int int_return = MPI_SUCCESS;
    if (rank)
    {
        int_return = MPI_Recv(u_k[0] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank - 1, 0, MPI_COMM_WORLD);
        err_if(int_return != MPI_SUCCESS, err_0);
    }
    if (rank + 1 < size)
    {
        int_return = MPI_Recv(u_k[rows - 1] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD);
        err_if(int_return != MPI_SUCCESS, err_0);
    }

err_0:
    return int_return;
}

static int file_write(const int size, const int rank,
    const size_t rows, const size_t m_y, dbl (* const restrict u_k)[m_y],
    const char * const restrict filename)
{
    int int_return = MPI_SUCCESS;
    for (size_t i = 1; i < m_x - 1; ++i)
    {
        size_t size_return = fwrite(u_k[i] + 1, sizeof(dbl), m_y - 2U, stream);
        err_not_eq(size_return, m_y - 2U, err_1);
    }
}
