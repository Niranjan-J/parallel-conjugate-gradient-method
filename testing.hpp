#include "linearalg.hpp"

#define element_limit 10

std::chrono::steady_clock::time_point START = std::chrono::steady_clock::now();
std::chrono::steady_clock::time_point END = std::chrono::steady_clock::now();

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

vector<double> solver(vector<double> &A, vector<int> &iA, vector<int> &jA, vector<double> &b, bool describe = false, bool assume_psd = true, const int iterations = 10, const double epsilon = 1e-5)
{
    int n = iA.size() - 1;
    vector<double> x_out(n);
    // clock_t start, end;

    vector<double> x0(n, 1);

    if (assume_psd)
    {
        START = std::chrono::steady_clock::now();
        parallel_Conjugate_Gradient(x_out, A, iA, jA, b, x0, iterations, epsilon);
        END = std::chrono::steady_clock::now();
    }
    else
    {

        vector<double> A_T(A.size());
        vector<int> A_Trow(n + 1);
        vector<int> A_Tcol(A.size());

        sparse_matrix_transpose(A, iA, jA, A_T, A_Trow, A_Tcol);

        if (!is_symmetric(A, A_T))
        {

            //R=A^T*A (n x n)
            vector<double> R;
            vector<int> Rrow(n + 1);
            vector<int> Rcol;

            A_TA(A_T, A_Trow, A_Tcol, R, Rrow, Rcol);

            //q=A^T*b (n x 1)
            vector<double> q(n);

            MatVecMult(q, A_T, A_Trow, A_Tcol, b);

            //Solve
            START = std::chrono::steady_clock::now();
            parallel_Conjugate_Gradient(x_out, R, Rrow, Rcol, q, x0, iterations, epsilon);

            END = std::chrono::steady_clock::now();
        }
        else
        {

            START = std::chrono::steady_clock::now();
            parallel_Conjugate_Gradient(x_out, A, iA, jA, b, x0, iterations, epsilon);

            END = std::chrono::steady_clock::now();
        }
    }

    if (describe)
    {
        cout << "Solution x :\n";
        for (int i = 0; i < x_out.size(); i++)
            cout << x_out[i] << endl;
    std::cout << "\nExecution time (microsec) = " << (std::chrono::duration_cast<std::chrono::microseconds>(END - START).count()) << std::endl;

        vector<double> res(n);
        MatVecMult(res, A, iA, jA, x_out, false);

        cout << "original b is:" << endl;
        for (auto u : b)
        {
            cout << u << " ";
        }
        cout << endl
             << "b_res is:" << endl;
        for (auto u : res)
        {
            cout << u << " ";
        }
        cout << endl;
    }

    return x_out;
}