#include "linearalg.hpp"

#define element_limit 10

vector<double> generate_random_b(vector<double> &A, vector<int> &iA, vector<int> &jA)
{
    int n = iA.size() - 1;
    vector<double> b(n), x(n);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        x[i] = rand() % 100;
    }

    MatVecMult(b, A, iA, jA, x, false);
    return b;
}

vector<vector<double>> generate_random_matrix(const int n = 4, const double sparse_proportion = 0.5)
{
    srand(time(NULL));

    assert((sparse_proportion < 1) && (sparse_proportion > 0));

    vector<vector<double>> A(n, vector<double>(n));

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {

            if (rand() < RAND_MAX * sparse_proportion)
            {
                A[i][j] = 0;
            }
            else
            {
                A[i][j] = rand() % element_limit; //make sure all elements are less than 10, so that we can make it diagonally dominant later
            }
        }
    }

    return A;
}

tuple<vector<double>, vector<int>, vector<int>> sparsify(const vector<vector<double>> &M)
{

    int n = M.size();

    vector<double> A;
    vector<int> IA = {0};
    vector<int> JA;
    int NNZ = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (M[i][j] != 0)
            {
                A.push_back(M[i][j]);
                JA.push_back(j);

                NNZ++;
            }
        }
        IA.push_back(NNZ);
    }
    return make_tuple(A, IA, JA);
}

vector<vector<double>> generate_random_symmetric_pd_matrix(const int n = 4, const double sparse_proportion = 0.5)
{
    vector<vector<double>> A = generate_random_matrix(n, sparse_proportion);

    //make it symmetric, A_new=(A+A_T)
#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (j < i)
            {

                A[i][j] += A[j][i];
                A[j][i] = A[i][j];
            }
        }
    }
    // make it diagonally dominant, A_new=A+2*n*I
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {

        A[i][i] += element_limit * n * 2;
    }

    //symmetric diagonally dominant matrix is positive definite

    return A;
}

pair<size_t, size_t> tester(int size, bool assume_psd = true, int iterations = 100,bool describe=false)
{
    if (assume_psd)
    {
        auto [A, iA, jA] = sparsify(generate_random_symmetric_pd_matrix(size));
        vector<double> b = generate_random_b(A, iA, jA);

        solver(A, iA, jA, b, describe, true, iterations);
    }
    else
    {
        auto [A, iA, jA] = sparsify(generate_random_matrix(size));
        vector<double> b = generate_random_b(A, iA, jA);

        solver(A, iA, jA, b, describe, false, iterations);
    }

    return {true,true};
}