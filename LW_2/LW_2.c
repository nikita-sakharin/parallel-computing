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

#define EPSILON (0.000244140625)
#define DEL_X (0.5)
#define DEL_Y (0.25)
#define FOUR (4U)

void start(int, int, size_t, dbl, size_t, dbl, dbl, const char * restrict);

static void init(int, int, size_t, size_t,
    dbl (* restrict * restrict)[], dbl (* restrict * restrict)[],
    dbl (* restrict * restrict)[], dbl * restrict * restrict);
static int send_recv_request(int, int, size_t, size_t,
    dbl (* restrict)[], dbl (* restrict)[], MPI_Request (* restrict)[FOUR]);
static void stratall_waitall(int, MPI_Request * restrict);
static dbl norm(int, int, size_t, size_t, dbl (* restrict u_k)[],
    dbl (* restrict u_k_plus_1)[], dbl * restrict);
static void finalize(size_t, dbl (* restrict)[], dbl (* restrict)[],
    dbl (* restrict)[], dbl * restrict);
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

    start(size, rank, m_x, DEL_X, m_y, DEL_Y, EPSILON, argv[3]);

    MPI_Finalize();

    return 0;
}

void start(const int size, const int rank,
    const size_t m_x, const dbl del_x, const size_t m_y, const dbl del_y,
    const dbl epsilon, const char * const restrict filename)
{
    mpi_abort_if(m_x < 3U || del_x < DBL_EPSILON || isinf(del_x) || isnan(del_x)
        || m_y < 3U || del_y < DBL_EPSILON || isinf(del_y) || isnan(del_y)
        || size < 1 || rank < 0 || size <= rank || m_x - 2U < (size_t) size,
        "m_x, del_x, m_y, del_y, size, rank");

    const size_t rows = (m_x - 2U) / size + ((size_t) rank < (m_x - 2U) % size) + 2U;
    dbl (* restrict u_k)[m_y], (* restrict u_k_plus_1)[m_y], (* restrict r)[m_y],
        * restrict all_norm;
    init(size, rank, rows, m_y, &u_k, &u_k_plus_1, &r, &all_norm);
    const dbl rdx2 = 1.0 / del_x / del_x, rdy2 = 1.0 / del_y / del_y,
              beta = 1.0 / (2.0 * (rdx2 + rdy2));
    const size_t rows_minus_1 = rows - 1,
                 m_y_minus_1 = m_y - 1;
    MPI_Request request[2U][FOUR];
    int count = send_recv_request(size, rank, rows, m_y, u_k, u_k_plus_1, request);
    dbl curr_norm = epsilon, wtime_start = MPI_Wtime(), wtime_end;
    for (size_t k = 0; curr_norm >= epsilon; ++k)
    {
        stratall_waitall(count, request[k % 2]);
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
        curr_norm = norm(size, rank, rows_minus_1, m_y_minus_1, u_k, u_k_plus_1, all_norm);
    }

    wtime_end = MPI_Wtime();
    printf("wtime = %lf\n", wtime_end - wtime_start);

    file_write(size, rank, rows, m_y, u_k, filename);
    finalize(m_y, u_k, u_k_plus_1, r, all_norm);
}

static void init(const int size, const int rank,
    const size_t rows, const size_t m_y,
    dbl (* restrict * const restrict u_k)[m_y],
    dbl (* restrict * const restrict u_k_plus_1)[m_y],
    dbl (* restrict * const restrict r)[m_y],
    dbl * restrict * restrict all_norm)
{
    *u_k        = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    *u_k_plus_1 = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    *r          = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    *all_norm   = (dbl *) malloc(size * sizeof(dbl));
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

static int send_recv_request(const int size, const int rank,
    const size_t rows, const size_t m_y,
    dbl (* const restrict u_k)[m_y], dbl (* const restrict u_k_plus_1)[m_y],
    MPI_Request (* const restrict request)[FOUR])
{
    int int_return, i = 0, j = 0;
    if (rank)
    {
        int_return = MPI_Send_init(u_k[1] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank - 1, 0, MPI_COMM_WORLD, request[0] + i++);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Send_init()");
        int_return = MPI_Recv_init(u_k[0] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank - 1, 0, MPI_COMM_WORLD, request[0] + i++);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Recv_init()");

        int_return = MPI_Send_init(u_k_plus_1[1] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank - 1, 0, MPI_COMM_WORLD, request[1] + j++);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Send_init()");
        int_return = MPI_Recv_init(u_k_plus_1[0] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank - 1, 0, MPI_COMM_WORLD, request[1] + j++);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Recv_init()");
    }

    if (rank + 1 < size)
    {
        int_return = MPI_Send_init(u_k[rows - 2U] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD, request[0] + i++);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Send_init()");
        int_return = MPI_Recv_init(u_k[rows - 1] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD, request[0] + i++);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Recv_init()");

        int_return = MPI_Send_init(u_k_plus_1[rows - 2U] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD, request[1] + j++);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Send_init()");
        int_return = MPI_Recv_init(u_k_plus_1[rows - 1] + 1, (int) m_y - 2, MPI_DOUBLE,
            rank + 1, 0, MPI_COMM_WORLD, request[1] + j++);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Recv_init()");
    }

    return i;
}

static void stratall_waitall(const int count, MPI_Request * restrict request)
{
    MPI_Status status[FOUR];
    int int_return = MPI_Startall(count, request);
    mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Startall()");
    int_return = MPI_Waitall(count, request, status);
    mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Waitall()");
    for (int i = 0; i < count; ++i)
    {
        mpi_abort_if(status[i].MPI_ERROR != MPI_SUCCESS, "MPI_Waitall()");
    }
}

static dbl norm(const int size, const int rank, const size_t rows, const size_t m_y,
    dbl (* const restrict u_k)[m_y], dbl (* const restrict u_k_plus_1)[m_y],
    dbl * const restrict all_norm)
{
    (void) rank;
    dbl curr_norm = 0.0;
    const size_t rows_minus_1 = rows - 1, m_y_minus_1 = m_y - 1;
    for (size_t i = 1; i < rows_minus_1; ++i)
    {
        for (size_t j = 1; j < m_y_minus_1; ++j)
        {
            curr_norm = max(curr_norm, fabs(u_k[i][j] - u_k_plus_1[i][j]));
        }
    }

    MPI_Allgather(&curr_norm, 1, MPI_DOUBLE, all_norm, 1, MPI_DOUBLE,
        MPI_COMM_WORLD);
    for (int i = 0; i < size; ++i)
    {
        curr_norm = max(curr_norm, all_norm[i]);
    }

    return curr_norm;
}

static void finalize(const size_t m_y, dbl (* const restrict u_k)[m_y],
    dbl (* const restrict u_k_plus_1)[m_y],
    dbl (* const restrict r)[m_y], dbl * const restrict all_norm)
{
    free(u_k);
    free(u_k_plus_1);
    free(r);
    free(all_norm);
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
