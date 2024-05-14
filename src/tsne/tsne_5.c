#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <stdint.h>
#include <immintrin.h>

#include "tsne.h"

static double h_beta(double *D, double *P, int n, int d, double beta, int exclude_index)
{
    double sumP = 0.0;
    double sumDP = 0.0;
    
    for (int i = 0; i < n; i++)
    {
        if (i == exclude_index)
        {
            P[i] = 0.0;
            continue;
        }

        P[i] = exp(-D[i] * beta); // use this? https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=exponential&ig_expand=748,2811,2816
        sumP += P[i];
        sumDP += D[i] * P[i];
    }
    
    double inv_sumP = 1.0 / sumP;
    __m256d vec_inv_sumP = _mm256_set1_pd(inv_sumP);
    
    int i = 0;
    __m256d slice_P;
    for (; i <= n - 4; i += 4) { // alignment? 
        slice_P = _mm256_loadu_pd(&P[i]);
        slice_P = _mm256_mul_pd(slice_P, vec_inv_sumP);
        _mm256_storeu_pd(&P[i], slice_P);
    }

    for (; i < n - 4; i += 4) { 
        __m256 inv_sumP_v = _mm256_set1_ps(inv_sumP);
        __m256* P_v = (__m256*)&P[i];
        *P_v = _mm256_mul_ps(*P_v, inv_sumP_v);
    }

    double di = sumDP / sumP;
    double l = log(sumP);
    double f = beta * di;
    double result = l + f;
    return result;
}




// W(n,dim_y) = 3n^2*dim_y
static void distance_squared(double *X, double *D, int n, int d)
{
    int k;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double dist = 0.0;
            double diff;

            // compute squared euclidean distance
            // vectorized with AVX
            __m256d vec_dist = _mm256_setzero_pd();
            __m256d x_slice, y_slice, vec_diff, vec_sq_diff;
            for (k = 0; k <= d - 4; k += 4) { // alignment?
                x_slice = _mm256_loadu_pd(&X[i * d + k]);
                y_slice = _mm256_loadu_pd(&X[j * d + k]);
                vec_diff = _mm256_sub_pd(x_slice, y_slice);
                vec_sq_diff = _mm256_mul_pd(vec_diff, vec_diff);
                vec_dist = _mm256_add_pd(vec_dist, vec_sq_diff);
            }
            
            // handle residual elements that cannot be vectorized
            double residual_dist = 0.0;
            for (; k < d; k++) {
                diff = X[i * d + k] - X[j * d + k];
                residual_dist += diff * diff;
            }

            // sum up vectorized and residual elements
            __m256d vec_sum = _mm256_hadd_pd(vec_dist, vec_dist);
            double final_dist = residual_dist + ((double*)&vec_sum)[0] + ((double*)&vec_sum)[2];
            D[i * n + j] = final_dist;
        }
    }
}


static void x2p(double *X, double *P, int n, int d, double perplexity)  //last function to vectorize
{
    double *D = (double *) malloc(sizeof(double) * n * n);
    double *beta = (double *) malloc(sizeof(double) * n);

    distance_squared(X, D, n, d);

    double logU = log(perplexity);

    for (int i = 0; i < n; i++)
    {
        beta[i] = 1.0;
        int m = i * n;
        double h = h_beta(D + m, P + m, n, d, beta[i], i);
        double Hdiff = h - logU;
        int tries = 0;
        double tol = 1e-5;

        double betamin = -INFINITY;
        double betamax = INFINITY;

        while (fabs(Hdiff) > tol && tries < 50) {
            double new_beta;
            if (Hdiff > 0) {
                betamin = beta[i];
                new_beta = (betamax == INFINITY || betamax == -INFINITY) ? beta[i] * 2.0 : (beta[i] + betamax) * 0.5;
            } else {
                betamax = beta[i];
                new_beta = (betamin == INFINITY || betamin == -INFINITY) ? beta[i] * 0.5 : (beta[i] + betamin) * 0.5;
            }

            h = h_beta(D + m, P + m, n, d, new_beta, i);
            Hdiff = h - logU;
            beta[i] = new_beta;
            tries += 1;
        }
    }
}


static void calculate_p(double *X, double *P, int n, int dx, double perplexity)
{
    x2p(X, P, n, dx, perplexity);

    for (int i = 0; i < n; i++)
    {
        for (int j = i+1; j < n; j++)
        {
            double tmp = (P[i * n + j] + P[j * n + i]) / (2*n);
            tmp = fmax(tmp, 1e-12);

            P[i * n + j] = tmp;
            P[j * n + i] = tmp;
        }
    }

     // Reorganize 
    int k = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = i+1; j < n; j++)
        {
            P[k] = P[i*n + j];
            k++;
        }
    }
}

// W(n) = 12n^2
static void compute_gradient(double *P, double *Q, double *Y, double *dY, int n, int dy)
{
    // 1. Compute the sum of Q
    double Qsum = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = i+1; j < n; j++)
        {
            double y_diff0 = Y[2*i + 0] - Y[2*j + 0];
            double y_diff1 = Y[2*i + 1] - Y[2*j + 1];
            double y_dist =  y_diff0 * y_diff0 + y_diff1 * y_diff1;

            Qsum += 2 / (1 + y_dist);
        }
    }
    double Qsum_inv = 1 / Qsum;

    memset(dY, 0, 2*n*sizeof(double));

    // 2. Update the gradients
    int P_idx = 0;
    for (int i = 0; i < n; i++)
    {
        double dYi0 = dY[2*i + 0];
        double dYi1 = dY[2*i + 1];

        for (int j = i+1; j < n; j++)
        {
            double y_diff0 = Y[2*i + 0] - Y[2*j + 0];
            double y_diff1 = Y[2*i + 1] - Y[2*j + 1];
            double y_dist =  y_diff0 * y_diff0 + y_diff1 * y_diff1;
            double y_dist_p1_inv = 1 / (1 + y_dist);

            double P_ij = P[P_idx];
            double Q_ij = Qsum_inv * y_dist_p1_inv;
            double PQ_diff = P_ij - Q_ij;
            double PQ_diff_times_p1_inv = PQ_diff*y_dist_p1_inv;

            double d1 = y_diff0 * PQ_diff_times_p1_inv;
            double d2 = y_diff1 * PQ_diff_times_p1_inv;

            dYi0 += d1;
            dYi1 += d2;

            dY[2*j + 0] -= d1;
            dY[2*j + 1] -= d2;

            P_idx++;
        }

        dY[2*i + 0] = dYi0;
        dY[2*i + 1] = dYi1;

    }
}

// W(n, dim_y) = n*dim_y
static void update_gains(double *gains, double *iY, double *dY, int n, int dy)
{
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < dy; k++)
        {
            if ((dY[i * dy + k] > 0.0) != (iY[i * dy + k] > 0.0))
            {
                gains[i * dy + k] += 0.2;
            }
            else
            {
                gains[i * dy + k] *= 0.8;
            }
        }
    }
}

// Does one step of gradient descent.
// W(n,dim_y) = 5*n*dim_y
static void update(double *Y, double *iY, double *dY, double *gains, int iter, int n, int dy, double eta, double momentum)
{
    update_gains(gains, iY, dY, n, dy);

    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < dy; k++)
        {
            iY[i * dy + k] = iY[i * dy + k] * momentum - eta * gains[i * dy + k] * dY[i * dy + k];
            Y[i * dy + k] += iY[i * dy + k];
        }
    }
}

// Centers Y to have a mean of 0.
// W(n,dim_y) = 2*n*dim_y
static void center_y(double *Y, int n, int dy)
{
    double *Y_SUM = (double*) malloc (sizeof(double) * n * dy);

    for (int i = 0; i < dy; i++)
    {
        Y_SUM[i] = 0.0;
        for (int j = 0; j < n; j++)
        {
            Y_SUM[i] += Y[j*dy+i];
        }
    }

    for (int k = 0; k < dy; k++)
    {
        Y_SUM[k] /= n;
    }
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < dy; k++)
        {
            Y[i * dy + k] -= Y_SUM[k];
        }
    }

    free(Y_SUM);
}

// Stops early exaggeration which divides all p's by 4.
static void multiply_p(double *P, int n, double factor) {
    int remainder = n % 4;
    int limit = n - remainder;
    __m256d factor_vec = _mm256_set1_pd(factor);
    __m256d one_vec = _mm256_set1_pd(1.0);

    for (int i = 0; i < n; i++) {
        int m = i * n;

        for (int j = 0; j < limit; j += 4) {
            int idx1 = m + j;
            __m256d p_values = _mm256_loadu_pd(&P[idx1]);
            __m256d mask = _mm256_cmp_pd(_mm256_set1_pd(i), p_values, _CMP_NEQ_OQ);
            __m256d new_values = _mm256_blendv_pd(p_values, _mm256_mul_pd(p_values, factor_vec), mask);
            _mm256_storeu_pd(&P[idx1], new_values);
        }

        for (int j = limit; j < n; j++) {
            int idx = m + j;
            P[idx] *= (i != j) ? factor : 1.0;
        }
    }
}


uint64_t tsne_5_flops(tsne_data input, tsne_hyperparameters hyperparams) {
    uint64_t n = input.n;
    uint64_t iter = hyperparams.iterations;

    return iter * 12*n*n;
}

void tsne_5(tsne_data input, tsne_hyperparameters hyperparams)
{
    double *X = input.input;
    double *Y = input.output;
    int n = input.n;
    int dx = input.input_dim;
    int dy = input.output_dim;

    // Probabilities p_ij
    double *P = (double *)malloc(sizeof(double) * n * n);
    // Probabilities q_ij
    double *Q = (double *)malloc(sizeof(double) * n * n);
    // Euclidean distances between Y's
    double *Y_distances = (double *)malloc(sizeof(double) * n * n);
    // Gains used for gradient descent
    double *gains = (double *)malloc(sizeof(double) * n * dy);
    // Previous gradient update
    double *iY = (double *)malloc(sizeof(double) * n * dy);
    // Current gradient update
    double *dY = (double *)malloc(sizeof(double) * n * dy);

    // Initialize some of the arrays
    for (int i = 0; i < n; i++)
    {
        for (int k = 0; k < dy; k++)
        {
            gains[i * dy + k] = 1.0;
            iY[i * dy + k] = 0.0;
            dY[i * dy + k] = 0.0;
        }
    }

    // Calculate p_ij
    calculate_p(X, P, n, dx, hyperparams.perplexity);
    multiply_p(P, n, 4.0); // Start early exaggeration

    for (int iter = 0; iter < hyperparams.iterations; iter++)
    {
        // Calculate gradients
        compute_gradient(P, Q, Y, dY, n, dy);

        // Update our state with the calculated gradients
        double momentum = 0.0;
        if (iter < 20)
        {
            momentum = hyperparams.initial_momentum;
        }
        else
        {
            momentum = hyperparams.final_momentum;
        }
        update(Y, iY, dY, gains, iter, n, dy, hyperparams.eta, momentum);

        // Center y (mean of 0)
        center_y(Y, n, dy);

        // Stop early exaggeration
        if (iter == 100)
        {
            multiply_p(P, n, 1.0/4.0);
        }
    }

    free(P);
    free(Q);
    free(Y_distances);
    free(gains);
    free(iY);
    free(dY);
}