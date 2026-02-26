#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>

/*
 * 1D horizon function (forward or backward direction)
 */
int horizon_1d(
    int n,         /* length of vectors z and h */
    double *z,     /* elevation function */
    int *h,        /* horizon function (return) */
    bool forward   /* direction: true for forward, false for backward */
)
{
    int i, k;

    // Set boundary condition
    if (forward) {
        h[n - 1] = n - 1;
    } else {
        h[0] = 0;
    }

    /*
     * Main loop. Parallelized only if not already inside a parallel region
     * (e.g., when called for a single row).
     */
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (i = 0; i < n; i++) {
        // Skip boundary point based on direction
        if (forward && i == n - 1) continue;
        if (!forward && i == 0) continue;

        double zi = z[i];
        double max_slope = 0.0;
        int max_point = i;

        if (forward) {
            for (k = i + 1; k < n; k++) {
                if (z[k] > zi) {
                    double dist = (double)(k - i);
                    double slope = (z[k] - zi) / dist;
                    if (slope > max_slope) {
                        max_slope = slope;
                        max_point = k;
                    }
                }
            }
        } else {
            for (k = i - 1; k >= 0; k--) {
                if (z[k] > zi) {
                    double dist = (double)(i - k);
                    double slope = (z[k] - zi) / dist;
                    if (slope > max_slope) {
                        max_slope = slope;
                        max_point = k;
                    }
                }
            }
        }
        h[i] = max_point;
    }
    return 0;
}


/*
**	Calculate values of cosines of angles to horizons, measured
**	from zenith, from elevation difference and distance.  Let
**	H be the angle from zenith and note that:
**
**		cos H = z / sqrt( z^2 + dis^2);
*/
void horizon_cos(
    int n,        /* length of horizon vector */
    double *z,    /* elevations */
    double delta, /* spacing */
    int *h,       /* horizon function */
    double *hcos  /* cosines of angles to horizon */
)
{
    int i;
    const double delta_sq = delta * delta;

    /*
     * Only parallelize if we aren't already in a parallel region
     * (e.g., when called for a single row instead of the whole 2D grid)
     */
    #pragma omp parallel for schedule(static) if(!omp_in_parallel())
    for (i = 0; i < n; ++i)
    {
        int j = h[i];

        // Point is its own horizon
        if (j == i)
        {
            hcos[i] = 0.0;
            continue;
        }

        double diff = z[j] - z[i];

        /*
         * Distance in grid cells. We square it immediately,
         * so the sign (direction) doesn't matter.
         */
        double d_idx = (double)(j - i);
        double dist_sq = (d_idx * d_idx) * delta_sq;

        /*
         * cos H = z / sqrt( z^2 + dist^2 )
         */
        hcos[i] = diff / sqrt(diff * diff + dist_sq);
    }
}


/*
 * Optimized 2D horizon function
 */
void horizon_2d(
    int nrows,    /* rows of elevations array */
    int ncols,    /* columns of elevations array */
    double *z,    /* elevations */
    double delta, /* spacing */
    bool forward, /* forward direction flag */
    double *hcos  /* cosines of angles to horizon */
)
{
    /*
     * Parallelize over rows. Each thread gets its own small horizon buffer
     * to avoid memory allocation inside the loop and eliminate redundant copies.
     */
    #pragma omp parallel
    {
        int *hbuf = (int *)malloc(ncols * sizeof(int));

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < nrows; i++)
        {
            double *z_row = &z[i * ncols];
            double *hcos_row = &hcos[i * ncols];

            // Calculate horizon indices for this row
            horizon_1d(ncols, z_row, hbuf, forward);

            // Compute cosine values directly into the output array
            horizon_cos(ncols, z_row, delta, hbuf, hcos_row);
        }

        free(hbuf);
    }
}
