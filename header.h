#ifndef __HEADER_H__
#define __HEADER_H__

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#define err_if(cnd_value, label)\
    if (cnd_value)\
    {\
        goto label;\
    }

#define exit_if(cnd_value, msg)\
    if (cnd_value)\
    {\
        if (errno)\
            perror(msg);\
        else\
            fprintf(stderr, "%s\n",  msg);\
        exit(EXIT_FAILURE);\
    }

#define mpi_abort_if(cnd_value, msg)\
    if (cnd_value)\
    {\
        if (errno)\
            perror(msg);\
        else\
            fprintf(stderr, "%s\n",  msg);\
        fflush(stderr);\
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_LASTCODE);\
    }

typedef signed char schar;
typedef unsigned char uchar;
typedef short shrt;
typedef unsigned short ushrt;
typedef unsigned uint;
typedef unsigned long ulong;
typedef long long llong;
typedef unsigned long long ullong;

typedef float flt;
typedef double dbl;
typedef long double ldbl;

#define max(a, b) ((a) >= (b) ? (a) : (b))
#define min(a, b) ((a) <= (b) ? (a) : (b))

#endif
