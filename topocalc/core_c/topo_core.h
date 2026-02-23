#ifndef TOPO_CORE_H
#define TOPO_CORE_H

#include <stdbool.h>

int hor1d(int n, double *z, int *h, bool forward);
void horval(int n, double *z, double delta, int *h, double *hcos);
void hor2d(int nrows, int ncols, double *z, double delta, bool forward, double *hcos);

#endif
