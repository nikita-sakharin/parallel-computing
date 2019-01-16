#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>

#include "../header.h"

#define EPSILON (0.000244140625)
#define DEL_X (0.5)
#define DEL_Y (0.25)

void start(size_t, dbl, size_t, dbl, dbl, const char * restrict);

static void init(size_t, size_t, dbl (* restrict * restrict)[],
    dbl (* restrict * restrict)[]);
static void finalize(void * restrict, void * restrict);
static void file_write(size_t, size_t, dbl (* restrict)[], const char * restrict);

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <m rows> <m columns> <file out>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    const ptrdiff_t m_x = strtoll(argv[1], NULL, 10) + 2,
        m_y = strtoll(argv[2], NULL, 10) + 2;
    exit_if(m_x < 0 || m_y < 0, "m_x < 0 || m_y < 0");

    start(m_x, DEL_X, m_y, DEL_Y, EPSILON, argv[3]);

    return 0;
}

void start(const size_t m_x, const dbl del_x, const size_t m_y, const dbl del_y,
    const dbl epsilon, const char * const restrict filename)
{
    exit_if(m_x < 3U || del_x < DBL_EPSILON || isinf(del_x) || isnan(del_x)
        || m_y < 3U || del_y < DBL_EPSILON || isinf(del_y) || isnan(del_y)
        || epsilon < DBL_EPSILON || isinf(epsilon) || isnan(epsilon),
        "m_x, del_x, m_y, del_y");

    dbl (* restrict u_k)[m_y], (* restrict r)[m_y];
    init(m_x, m_y, &u_k, &r);
    const dbl rdx2 = 1.0 / del_x / del_x, rdy2 = 1.0 / del_y / del_y,
              beta = 1.0 / (2.0 * (rdx2 + rdy2));
    const size_t m_y_minus_1 = m_y - 1;
    dbl norm[2U] = { epsilon }, wtime_start = omp_get_wtime(), wtime_end;
#   pragma omp parallel
    {
        const size_t size = omp_get_num_threads(), rank = omp_get_thread_num(),
            begin = rank * (m_x - 2U) / size + 1,
            end = (rank + 1) * (m_x - 2U) / size + 1;
        for (size_t k = 0; max(norm[0], norm[1]) >= epsilon; k ^= 1)
        {
#           pragma omp barrier
#           pragma omp single
            {
                norm[k] = 0.0;
            }
            dbl norm_k = 0.0;
            for (size_t i = begin; i < end; ++i)
            {
                for (size_t j = 1 + (i + k) % 2U; j < m_y_minus_1; j += 2U)
                {
                    const dbl u_k_i_j = u_k[i][j];
                    u_k[i][j] = ((u_k[i - 1][j] + u_k[i + 1][j]) * rdx2 +
                        (u_k[i][j - 1] + u_k[i][j + 1]) * rdy2 - r[i][j]) * beta;
                    norm_k = max(norm_k, fabs(u_k_i_j - u_k[i][j]));
                }
            }
#           pragma omp critical
            {
                norm[k] = max(norm[k], norm_k);
            }
#           pragma omp barrier
        }
    }
    wtime_end = omp_get_wtime();
    printf("wtime = %lf\n", wtime_end - wtime_start);

    file_write(m_x, m_y, u_k, filename);
    finalize(u_k, r);
}

static void init(const size_t m_x, const size_t m_y,
    dbl (* restrict * const restrict u_k)[m_y],
    dbl (* restrict * const restrict r)[m_y])
{
    *u_k        = (dbl (*)[m_y]) malloc(m_x * m_y * sizeof(dbl));
    *r          = (dbl (*)[m_y]) malloc(m_x * m_y * sizeof(dbl));
    exit_if(!*u_k || !*r, "malloc()");

#   pragma omp parallel
    {
        const size_t size = omp_get_num_threads(), rank = omp_get_thread_num(),
            begin = rank * m_x / size,
            end = (rank + 1) * m_x / size;
        for (size_t i = begin; i < end; ++i)
        {
            for (size_t j = 0; j < m_y; ++j)
            {
                if (!i || i + 1 == m_x || !j || j + 1 == m_y)
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
    }
}

static void finalize(void * const restrict u_k, void * const restrict r)
{
    free(u_k);
    free(r);
}

static void file_write(const size_t m_x, const size_t m_y,
    dbl (* const restrict u_k)[m_y], const char * const restrict filename)
{
    int int_return;
    FILE * const restrict file_out = fopen(filename, "wb");
    exit_if(!file_out, "fopen()");

    const size_t m_x_minus_1 = m_x - 1, m_y_minus_2 = m_y - 2U;
    for (size_t i = 1; i < m_x_minus_1; ++i)
    {
        size_t size_return = fwrite(u_k[i] + 1, sizeof(dbl), m_y_minus_2, file_out);
        exit_if(size_return != m_y_minus_2, "fwrite()");
    }
    int_return = fclose(file_out);
    exit_if(int_return == EOF, "fclose()");
}
