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
    
    double inv_sumP = 0.0;
    for (int i = 0; i < n; i++)
    {
        if (i == exclude_index)
        {
            P[i] = 0.0;
            continue;
        }

        P[i] = exp(-D[i] * beta);
        sumP += P[i];
        sumDP += D[i] * P[i];
    }

    if (sumP != 0.0) {
        inv_sumP = 1.0 / sumP;
    }

    for (int i = 0; i < n; i++)
    {
        P[i] *= inv_sumP; 
    }

    return log(sumP) + beta * sumDP * inv_sumP;
}

// W(n,dim_y) = 3n^2*dim_y
static void distance_squared(double *X, double *D, int n, int d)
{
    for (int i = 0; i < n; i++)
    {
        int q = i * d;
        int w = i * n;
        for (int j = 0; j < n; j++)
        {
            int t = j * d;
            double dist = 0.0;
            double diff;
            for (int k = 0; k < d; k++)
            {
                diff = X[q + k] - X[t + k];
                dist += diff * diff;
            }
            D[w + j] = dist;
        }
    }
}

static void x2p(double *X, double *P, int n, int d, double perplexity)
{
    double *D = (double *) malloc(sizeof(double) * n * n);
    double *beta = (double *) malloc(sizeof(double) * n);

    distance_squared(X, D, n, d);
    double tol = 1e-5;
    double logU = log(perplexity);

    for (int i = 0; i < n; i++)
    {
        beta[i] = 1.0;
        double h = h_beta(D + i * n, P + i * n, n, d, beta[i], i);
        double Hdiff = h - logU;
        int tries = 0;

        double betamin = -INFINITY;
        double betamax = INFINITY;

        while (fabs(Hdiff) > tol && tries < 50)
        {
            if (Hdiff > 0)
            {
                betamin = beta[i];
                if (betamax == INFINITY || betamax == -INFINITY)
                {
                    beta[i] = beta[i] * 2.;
                }
                else
                {
                    beta[i] = (beta[i] + betamax) / 2.;
                }
            }
            else
            {
                betamax = beta[i];
                if (betamin == INFINITY || betamin == -INFINITY)
                {
                    beta[i] = beta[i] / 2.;
                }
                else
                {
                    beta[i] = (beta[i] + betamin) / 2.;
                }
            }

            h = h_beta(D + i * n, P + i * n, n, d, beta[i], i);
            Hdiff = h - logU;
            tries += 1;
        }
    }

    free(D);
    free(beta);
}

static void calculate_p(double *X, double *P, int n, int dx, double perplexity)
{
    x2p(X, P, n, dx, perplexity);
    int nn = 2 * n;
    double inv_nn = 1.0 / nn;

    for (int i = 0; i < n; i++)
    {
        int q = i * n;
        for (int j = i + 1; j < n; j++)
        {
            int w = j * n;
            int wi = w + i;
            int qj = q + j;

            double tmp_sum = P[qj] + P[wi];
            double tmp = tmp_sum * inv_nn;

            P[qj] = tmp;
            P[wi] = tmp;
        }
    }
}


// W(n) = 18n^2
static void compute_gradient(double *P, double *Q, double *Y, double *dY, int n, int dy)
{
    // 1. Compute the sum of Q
    double Qsum = 0.0;
    for (int i = 0; i < n; i++)
    {
        int q = 2 * i;
        for (int j = i+1; j < n; j++)
        {
            int w = 2 * j;
            double y_diff0 = Y[q] - Y[w];
            double y_diff1 = Y[q + 1] - Y[w + 1];
            double a = y_diff0 * y_diff0;
            double b = y_diff1 * y_diff1;
            double y_dist = a + b;
            double inv = 1 + y_dist;
            Qsum += 2 / inv;
        }
    }
    double Qsum_inv = 1 / Qsum;

    // 2. Update the gradients
    for (int i = 0; i < n; i++)
    {
        int q = 2 * i;
        int r = i * dy;
        dY[r] = 0.0;
        dY[r + 1] = 0.0;
        for (int j = 0; j < n; j++)
        {
            int w = 2 * j;
            double y_diff0 = Y[q] - Y[w];
            double y_diff1 = Y[q + 1] - Y[w + 1];
            double a = y_diff0 * y_diff0;
            double b = y_diff1 * y_diff1;
            double y_dist = a + b;
            double inv = 1 + y_dist;
            double y_dist_p1_inv = 1 / inv;

            double P_ij = P[i * n + j];
            double Q_ij = i != j ? Qsum_inv * y_dist_p1_inv : 0.0;
            double PQ_diff = P_ij - Q_ij;
            double PQ_diff_times_p1_inv = PQ_diff*y_dist_p1_inv;

            dY[r] += y_diff0 * PQ_diff_times_p1_inv;
            dY[r + 1] += y_diff1 * PQ_diff_times_p1_inv;
        }
    }
}

// W(n, dim_y) = n*dim_y
static void update_gains(double *gains, double *iY, double *dY, int n, int dy)
{
    for (int i = 0; i < n; i++)
    {
        int q = i * dy;
        for (int k = 0; k < dy; k++)
        {
            int w = q + k;
            if ((dY[w] > 0.0) != (iY[w] > 0.0))
            {
                gains[w] += 0.2;
            }
            else
            {
                gains[w] *= 0.8;
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
        int q = i * dy;
        for (int k = 0; k < dy; k++)
        {
            int w = q + k;
            double a = iY[w] * momentum;
            iY[w] = a - eta * gains[w] * dY[w];
            Y[w] += iY[w];
        }
    }
}

// Centers Y to have a mean of 0.
// W(n,dim_y) = 2*n*dim_y
static void center_y(double *Y, int n, int dy)
{
    double *Y_SUM = (double*)malloc(sizeof(double) * dy);
    for (int i = 0; i < dy; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            int q = j * dy;
            sum += Y[q + i];
        }
        Y_SUM[i] = sum / n;
    }

    for (int i = 0; i < n; i++)
    {
        int w = i * dy;
        for (int k = 0; k < dy; k++)
        {
            Y[w + k] -= Y_SUM[k];
        }
    }

    free(Y_SUM);
}

// Stops early exaggeration which divides all p's by 4.
static void multiply_p(double *P, int n, double factor)
{
    int num_iters = n / 4;
    __m256d factor_avx = _mm256_set1_pd(factor);
    for (int i = 0; i < n; ++i)
    {
        int q = i * n;
        for (int j = 0; j < num_iters; ++j)
        {
            int w = j * 4;
            int m = q + w;
            __m256d P_avx = _mm256_loadu_pd(&P[m]);
            P_avx = _mm256_mul_pd(P_avx, factor_avx);
            _mm256_storeu_pd(&P[m], P_avx);
        }
    }
}

uint64_t tsne_5_flops(tsne_data input, tsne_hyperparameters hyperparams) {
    uint64_t n = input.n;
    uint64_t iter = hyperparams.iterations;

    return iter * 18*n*n;
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
        int q = i * dy;
        for (int k = 0; k < dy; k++)
        {
            int w = q + k;
            gains[w] = 1.0;
            iY[w] = 0.0;
            dY[w] = 0.0;
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
        // Set momentum based on the value of iter
        momentum = hyperparams.initial_momentum + (hyperparams.final_momentum - hyperparams.initial_momentum) * (iter >= 20);

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
