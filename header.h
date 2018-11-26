#ifndef __HEADER_H__
#define __HEADER_H__

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

#endif
