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

static int start(size_t, dbl, size_t, dbl, int, FILE * restrict);

int main(int argc, const char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <nrows> <ncolumns> <file out>\n", argv[0]);
        return -1;
    }

    FILE * const stream = fopen(argv[3], "wb");
    err_eq(stream, NULL, err_0);

    int int_return = start(atoi(argv[1]) + 2, DEL_X, atoi(argv[2]) + 2, DEL_Y,
                           N, stream);
    err_not_eq(int_return, 0, err_0);

    return 0;

err_0:
    perror("");
    exit(EXIT_FAILURE);
}

static int start(const size_t m_x, const dbl del_x, const size_t m_y, const dbl del_y,
    const int n, FILE * const restrict stream)
{
    int return_value = -1;
    dbl (* restrict u_k)[m_y],
        (* restrict u_k_plus_1)[m_y],
        (* restrict r)[m_y];

    err_if(!m_x || del_x < DBL_EPSILON || !m_y || del_y < DBL_EPSILON, err_0);

    const dbl rdx2 = 1.0 / del_x / del_x,
              rdy2 = 1.0 / del_y / del_y,
              beta = 1.0 / (2.0 * (rdx2 + rdy2));
    u_k        = (dbl (*)[m_y]) malloc(m_x * m_y * sizeof(dbl));
    u_k_plus_1 = (dbl (*)[m_y]) malloc(m_x * m_y * sizeof(dbl));
    r          = (dbl (*)[m_y]) malloc(m_x * m_y * sizeof(dbl));
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
        size_t size_return = fwrite(u_k[i] + 1, sizeof(dbl), m_y - 2, stream);
        err_not_eq(size_return, m_y - 2, err_1);
    }
    return_value = 0;

err_1:
    free(r);
    free(u_k_plus_1);
    free(u_k);
err_0:
    return return_value;
}
