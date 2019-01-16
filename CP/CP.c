#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "../header.h"

#define N (100U)
#define H_X (0.5)
#define H_Y (0.25)

#define TWO (2U)

size_t file_read(int, int, size_t (* restrict * restrict)[TWO],
    size_t (* restrict * restrict)[TWO], const char * restrict);
void start(int, int, size_t, const size_t (* restrict)[TWO],
    const size_t (* restrict)[TWO], dbl, dbl, uint, const char * restrict);

static void init(int, int, size_t, const size_t (* restrict)[TWO],
    const size_t (* restrict)[TWO], dbl * restrict * restrict * restrict,
    dbl * restrict * restrict * restrict);
static void finalize(size_t, const size_t (* restrict)[TWO],
    dbl * restrict * restrict, dbl * restrict * restrict);
static void sync(int, int, size_t, const size_t (* restrict)[TWO],
    dbl * const restrict * restrict);
static void sendrecv(int, int, size_t, const size_t (* restrict)[TWO],
    dbl * const restrict * restrict, bool);
static void file_write(int, int, size_t, const size_t (* restrict)[TWO],
    const dbl * restrict * restrict, const char * restrict);

static bool in_rangez(size_t, size_t, size_t);
static bool in_rangej(ptrdiff_t, ptrdiff_t, ptrdiff_t);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <file in> <file out>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_LASTCODE);
    }

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t (* restrict in_range_y)[TWO], (* restrict out_range_y)[TWO];
    const size_t m_x = file_read(size, rank, &in_range_y, &out_range_y, argv[1]);

    start(size, rank, m_x, (const size_t (*)[TWO]) in_range_y,
        (const size_t (*)[TWO]) out_range_y, H_X, H_Y, N, argv[2]);

    MPI_Finalize();

    free(in_range_y);
    free(out_range_y);

    return 0;
}

size_t file_read(const int size, const int rank,
    size_t (* restrict * const restrict in_range_y)[TWO],
    size_t (* restrict * const restrict out_range_y)[TWO],
    const char * const restrict filename)
{
    mpi_abort_if(rank < 0 || size <= rank, "size, rank");

    int int_return;
    FILE * restrict file_in = fopen(filename, "r");
    mpi_abort_if(!file_in, "fopen()");

    ptrdiff_t m_x, min_y = PTRDIFF_MAX;
    int_return = fscanf(file_in, "%jd", &m_x);
    mpi_abort_if(int_return != 1 || m_x < 1 || m_x < size, "fscanf(), m_x");

    const size_t begin_x = rank * m_x / size + 1, end_x = (rank + 1) * m_x / size + 1,
        rows = end_x - begin_x + 2U;
    *in_range_y = (size_t (*)[TWO]) malloc(rows * TWO * sizeof(size_t));
    *out_range_y = (size_t (*)[TWO]) malloc(rows * TWO * sizeof(size_t));
    mpi_abort_if(!*in_range_y || !*out_range_y, "malloc()");

    for (ptrdiff_t i = 0, j = rank == 0; i < m_x; ++i)
    {
        ptrdiff_t begin, end;
        int_return = fscanf(file_in, "%jd%jd", &begin, &end);
        mpi_abort_if(int_return != 2 || begin < 1 || begin >= end,
            "fscanf(), begin, end");
        if (in_rangej(i + 1, begin_x, end_x) ||
            (rank && (size_t) i + 2U == begin_x) ||
            (rank + 1 != size && (size_t) i + 1 == end_x))
        {
            (*in_range_y)[j][0] = begin;
            (*in_range_y)[j][1] = end;
            ++j;
        }
        min_y = min(min_y, begin);
    }
    mpi_abort_if(min_y != 1, "min_y");

    if (!rank)
    {
        (*in_range_y)[0][0] = (*in_range_y)[0][1] = SIZE_MAX;
        (*out_range_y)[0][0] = (*in_range_y)[1][0];
        (*out_range_y)[0][1] = (*in_range_y)[1][1];
    }
    if (rank + 1 == size)
    {
        (*in_range_y)[rows - 1][0] = (*in_range_y)[rows - 1][1] = SIZE_MAX;
        (*out_range_y)[rows - 1][0] = (*in_range_y)[rows - 2][0];
        (*out_range_y)[rows - 1][1] = (*in_range_y)[rows - 2][1];
    }
    for (size_t i = 0; i < rows; ++i)
    {
        if ((!rank && !i) || (rank + 1 == size && i + 1 == rows))
            continue;
        (*out_range_y)[i][0] = (*in_range_y)[i][0] - 1;
        (*out_range_y)[i][1] = (*in_range_y)[i][1] + 1;
        if (i > 1 || (rank && i))
        {
            (*out_range_y)[i][0] = min((*out_range_y)[i][0], (*in_range_y)[i - 1][0]);
            (*out_range_y)[i][1] = max((*out_range_y)[i][1], (*in_range_y)[i - 1][1]);
        }
        if (i + 2U < rows || (rank + 1 != size && i + 1 != rows))
        {
            (*out_range_y)[i][0] = min((*out_range_y)[i][0], (*in_range_y)[i + 1][0]);
            (*out_range_y)[i][1] = max((*out_range_y)[i][1], (*in_range_y)[i + 1][1]);
        }
    }

    int_return = fclose(file_in);
    mpi_abort_if(int_return == EOF, "fclose()")

    return (size_t) m_x + 2U;
}

void start(const int size, const int rank, const size_t m_x,
    const size_t (* const restrict in_range_y)[TWO],
    const size_t (* const restrict out_range_y)[TWO], const dbl h_x, const dbl h_y,
    const uint n, const char * const restrict filename)
{
    mpi_abort_if(m_x < 3 || rank < 0 || size <= rank || m_x - 2U < (size_t) size
        || h_x < DBL_EPSILON || isinf(h_x) || isnan(h_x)
        || h_y < DBL_EPSILON || isinf(h_y) || isnan(h_y) || !n,
        "m_x, h_x, h_y, n, size, rank");

    const size_t begin = rank * (m_x - 2U) / size + 1,
        end = (rank + 1) * (m_x - 2U) / size + 1,
    rows = end - begin + 2U, rows_minus_1 = rows - 1;
    dbl * restrict * restrict u_k, * restrict * restrict r;
    init(size, rank, rows, in_range_y, out_range_y, &u_k, &r);
    const dbl rdx2 = 1.0 / h_x / h_x, rdy2 = 1.0 / h_y / h_y,
              beta = 1.0 / (2.0 * (rdx2 + rdy2));
    for (uint k = 0; k < n; ++k)
    {
        for (size_t i = 1; i < rows_minus_1; ++i)
        {
            for (size_t j = in_range_y[i][0] + (in_range_y[i][0] + begin + i - 1 + k) % 2U;
                j < in_range_y[i][1]; j += 2U)
            {
                u_k[i][j] = ((u_k[i - 1][j] + u_k[i + 1][j]) * rdx2 +
                    (u_k[i][j - 1] + u_k[i][j + 1]) * rdy2 - r[i][j]) * beta;
            }
        }
        sync(size, rank, rows, in_range_y, u_k);
    }

    file_write(size, rank, rows, in_range_y, (const dbl **) u_k, filename);
    finalize(rows, out_range_y, u_k, r);
}

static void init(const int size, const int rank,
    const size_t rows, const size_t (* const restrict in_range_y)[TWO],
    const size_t (* const restrict out_range_y)[TWO],
    dbl * restrict * restrict * const restrict u_k,
    dbl * restrict * restrict * const restrict r)
{
    *u_k = (dbl **) malloc(rows * sizeof(dbl *)),
    *r   = (dbl **) malloc(rows * sizeof(dbl *));
    mpi_abort_if(!*u_k || !*r, "malloc()");
    for (size_t i = 0; i < rows; ++i)
    {
        (*u_k)[i] = malloc((out_range_y[i][1] - out_range_y[i][0]) * sizeof(dbl));
        (*r)[i]   = malloc((out_range_y[i][1] - out_range_y[i][0]) * sizeof(dbl));
        mpi_abort_if(!(*u_k)[i] || !(*r)[i], "malloc()");
        (*u_k)[i] -= out_range_y[i][0];
        (*r)[i]   -= out_range_y[i][0];
    }
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = out_range_y[i][0]; j < out_range_y[i][1]; ++j)
        {
            if ((!i && !rank) || (i + 1 == rows && rank + 1 == size) ||
                !in_rangez(j, in_range_y[i][0], in_range_y[i][1]))
                (*u_k)[i][j] = 1.0;
            else
                (*u_k)[i][j] = 0.0;
            (*r)[i][j] = 0.0;
        }
    }
}

static void finalize(const size_t rows,
    const size_t (* const restrict out_range_y)[TWO],
    dbl * restrict * const restrict u_k, dbl * restrict * const restrict r)
{
    for (size_t i = 0; i < rows; ++i)
    {
        free(u_k[i] + out_range_y[i][0]);
        free(r[i] + out_range_y[i][0]);
    }
    free((dbl **) u_k);
    free((dbl **) r);
}

static void sync(const int size, const int rank,
    const size_t rows, const size_t (* const restrict in_range_y)[TWO],
    dbl * const restrict * const restrict u_k)
{
    if (rank % 2)
    {
        sendrecv(size, rank, rows, in_range_y, u_k, true);
        sendrecv(size, rank, rows, in_range_y, u_k, false);
    }
    else
    {
        sendrecv(size, rank, rows, in_range_y, u_k, false);
        sendrecv(size, rank, rows, in_range_y, u_k, true);
    }
}

static void sendrecv(const int size, const int rank, const size_t rows,
    const size_t (* const restrict in_range_y)[TWO],
    dbl * const restrict * const restrict u_k, const bool direction)
{
    if ((direction && !rank) || (!direction && rank + 1 == size))
        return;

    int int_return;
    MPI_Status status;
    const ptrdiff_t i = (direction ? 0 : rows - 1), j = i + (direction ? 1 : -1);
    int_return = MPI_Sendrecv(
        u_k[j] + in_range_y[j][0], in_range_y[j][1] - in_range_y[j][0],
        MPI_DOUBLE, rank + (i - j), 0,
        u_k[i] + in_range_y[i][0], in_range_y[i][1] - in_range_y[i][0],
        MPI_DOUBLE, rank + (i - j), 0, MPI_COMM_WORLD, &status);
/*
    mpi_abort_if(int_return != MPI_SUCCESS || status.MPI_ERROR != MPI_SUCCESS,
        "MPI_Sendrecv()");
*/
    (void) int_return;
}

static void file_write(const int size, const int rank, const size_t rows,
    const size_t (* const restrict in_range_y)[TWO],
    const dbl * restrict * const restrict u_k,
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

    const size_t rows_minus_1 = rows - 1;
    for (int r = 0; r < size; ++r)
    {
        int_return = MPI_Barrier(MPI_COMM_WORLD);
        mpi_abort_if(int_return != MPI_SUCCESS, "MPI_Barrier()");
        if (r == rank)
        {
            file_out = fopen(filename, "ab");
            mpi_abort_if(!file_out, "fopen()");
            for (size_t i = 1; i < rows_minus_1; ++i)
            {
                const size_t diff = in_range_y[i][1] - in_range_y[i][0];
                size_t size_return = fwrite(u_k[i] + in_range_y[i][0],
                    sizeof(dbl), diff, file_out);
                mpi_abort_if(size_return != diff, "fwrite()");

            }
            int_return = fclose(file_out);
            mpi_abort_if(int_return == EOF, "fclose()");
        }
    }
}

inline static bool in_rangez(const size_t i,
    const size_t begin, const size_t end)
{
    return begin <= i && i < end;
}

inline static bool in_rangej(const ptrdiff_t i,
    const ptrdiff_t begin, const ptrdiff_t end)
{
    return begin <= i && i < end;
}
