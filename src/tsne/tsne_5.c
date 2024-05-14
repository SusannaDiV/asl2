#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <stdint.h>

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

        P[i] = exp(-D[i] * beta);
        sumP += P[i];
        sumDP += D[i] * P[i];
    }

    for (int i = 0; i < n; i++)
    {
        P[i] /= sumP; // WTF do this every time? We need only the final P
    }

    return log(sumP) + beta * sumDP / sumP;
}


// W(n,dim_y) = 3n^2*dim_y
static void distance_squared(double *X, double *D, int n, int d)
{
    for (int i = 0; i < n; i++)
    {
        int q = i * d;
        int m = i * n;
        for (int j = 0; j < n; j++)
        {
            int w = j * d;
            double dist = 0.0;
            double diff;
            for (int k = 0; k < d; k++)
            {
                diff = X[q + k] - X[w + k];
                dist += diff * diff;
            }
            D[m + j] = dist;
        }
    }
}

static void x2p(double *X, double *P, int n, int d, double perplexity)
{
    double *D = (double *) malloc(sizeof(double) * n * n);
    double *beta = (double *) malloc(sizeof(double) * n);

    distance_squared(X, D, n, d);

    double logU = log(perplexity);

    for (int i = 0; i < n; i++)
    {
        beta[i] = 1.0;
        int q = i * n;
        double h = h_beta(q+D, q+P, n, d, beta[i], i);
        double Hdiff = h - logU;
        int tries = 0;
        double tol = 1e-5;

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
                    beta[i] = (beta[i] + betamax) * 0.5;
                }
            }
            else
            {
                betamax = beta[i];
                if (betamin == INFINITY || betamin == -INFINITY)
                {
                    beta[i] = beta[i] * 0.5;
                }
                else
                {
                    beta[i] = (beta[i] + betamin) * 0.5;
                }
            }

            h = h_beta(q+D, q+P, n, d, beta[i], i);
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
static void multiply_p(double *P, int n, double factor)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            P[i * n + j] *= factor;
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