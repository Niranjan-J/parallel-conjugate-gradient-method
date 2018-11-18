#include "linearalg.hpp"
#include "testing.hpp"

int main()
{

    auto [A, iA, jA] = sparsify(generate_random_symmetric_pd_matrix(3));
    vector<double> b = generate_random_b(A, iA, jA);

    solver(A, iA, jA, b, true);

    cout << endl
         << "Now with a non psd matrix" << endl;

    tie(A, iA, jA) = sparsify(generate_random_matrix(3));
    b = generate_random_b(A, iA, jA); //TODO : random b

    solver(A, iA, jA, b, true, false, 100);

    return 0;
}
