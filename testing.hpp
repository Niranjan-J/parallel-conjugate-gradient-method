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



pair<double, double> tester(int size, bool assume_psd = true, int iterations = 100, bool describe = false,int repeats=10)
{
    std::chrono::steady_clock::time_point START = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point END = std::chrono::steady_clock::now();

    double new_time;
    double moving_avg = 0, prev_moving_avg = 0;
    double moving_std_dev = 0;

    auto [A, iA, jA] = sparsify(generate_random_symmetric_pd_matrix(0));

    if (assume_psd)
    {
        tie(A, iA, jA) = sparsify(generate_random_symmetric_pd_matrix(size));
    }
    else
    {
        tie(A, iA, jA) = sparsify(generate_random_matrix(size));
    }

    vector<double> b = generate_random_b(A, iA, jA);

    for (int i = 0; i < repeats; i++)

    {
        START = std::chrono::steady_clock::now();
        solver(A, iA, jA, b, describe, assume_psd, iterations);
        END = std::chrono::steady_clock::now();

        new_time=std::chrono::duration_cast<std::chrono::microseconds>(END - START).count();
        // cout<<new_time<<endl;

        moving_avg = moving_avg + (new_time - moving_avg) / (i + 1);
        moving_std_dev = moving_std_dev + (new_time - prev_moving_avg) * (new_time - moving_avg);
        prev_moving_avg=moving_avg;
    }

    return {moving_avg, sqrt(moving_std_dev/(repeats-1))};//sample std dev
}