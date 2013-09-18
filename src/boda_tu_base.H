// this file should be the first include at start of each translation unit. it should not be multiply included.
#ifdef BODA_TU_BASE_H
#error "boda_tu_base.h should only be include once at the start of each translation unit"
#else
#define BODA_TU_BASE_H
#include"Python.h"
#include"boda_base.H"

#endif // BODA_TU_BASE_H
