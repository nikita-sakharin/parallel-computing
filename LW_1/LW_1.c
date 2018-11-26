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
static int synchronize(int, int, size_t, size_t my, dbl (* restrict u_k)[m_y]);

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
    int return_value = -1;
    dbl (* restrict u_k)[m_y],
        (* restrict u_k_plus_1)[m_y],
        (* restrict r)[m_y];

    err_if(m_x < 2U || del_x < DBL_EPSILON || isinf(del_x) || isnan(del_x)
        || m_y < 2U || del_y < DBL_EPSILON || isinf(del_y) || isnan(del_y)
        || !size || rank < 0 || size <= rank, err_0);

    const size_t rows = (m_x - 2U) / size + (rank < (m_x - 2U) % size) + 2U;
    if (rows <= 2U)
    {
        return 0;
    }

    const dbl rdx2 = 1.0 / del_x / del_x,
              rdy2 = 1.0 / del_y / del_y,
              beta = 1.0 / (2.0 * (rdx2 + rdy2));
    u_k        = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    u_k_plus_1 = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    r          = (dbl (*)[m_y]) malloc(rows * m_y * sizeof(dbl));
    err_if(!u_k || !u_k_plus_1 || !r, err_1);

    for (size_t i = 0; i < m_x; ++i)
    {
        for (size_t j = 0; j < m_y; ++j)
        {
            if (i == 0 || j == 0 || i == (m_x - 1) || j == (m_y - 1))
            {
                u_k_plus_1[i][j] = u_k[i][j] = 1.0;
            }
            else
            {
                u_k_plus_1[i][j] = u_k[i][j] = 0.0;
            }
            r[i][j] = 0.0;
        }
    }

    const size_t m_x_minus_1 = m_x - 1,
                 m_y_minus_1 = m_y - 1;
    for (int k = 0; k < n; ++k)
    {
        synchronize(size, rank, rows, m_y, u_k);
        for (size_t i = 1; i < m_x_minus_1; ++i)
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

    for (size_t i = 1; i < m_x - 1; ++i)
    {
        size_t size_return = fwrite(u_k[i] + 1, sizeof(dbl), m_y - 2U, stream);
        err_not_eq(size_return, m_y - 2U, err_1);
    }
    return_value = 0;

err_1:
    free(r);
    free(u_k_plus_1);
    free(u_k);
err_0:
    return return_value;
}

static int synchronize(int size, int rank, const size_t rows, const size_t m_y,
    dbl (* restrict u_k)[m_y])
{
    (void) size;
    if (rank % 2U)
    {
        ;
    }
    else
    {
        ;
    }
}
