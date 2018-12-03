#include "my_library.hpp"

int main()
{

    ifstream input("input.txt");
    ofstream output("output.txt");

    int num_inputs;
    input >> num_inputs >> std::ws;

    for (int i = 0; i < num_inputs; i++)
    {
        string line;

        //Sparse Matrix A (n x n)
        vector<double> A;
        vector<int> iA;
        vector<int> jA;

        //b (n x 1)
        vector<double> b;

        {
            getline(input, line);
            stringstream numbers(line);
            copy(istream_iterator<double>(numbers),
                 istream_iterator<double>(),
                 back_inserter(A));
        }

        {
            getline(input, line);
            stringstream numbers(line);
            copy(istream_iterator<int>(numbers),
                 istream_iterator<int>(),
                 back_inserter(iA));
        }

        {
            getline(input, line);
            stringstream numbers(line);
            copy(istream_iterator<int>(numbers),
                 istream_iterator<int>(),
                 back_inserter(jA));
        }

        {
            getline(input, line);
            stringstream numbers(line);
            copy(istream_iterator<double>(numbers),
                 istream_iterator<double>(),
                 back_inserter(b));
        }

        int n = iA.size() - 1;

        //Initial x will be n 1's

        //Output x
        vector<double> x_out(n);

        //Number of iterations and accuracy
        int iterations = 10;
        double epsilon = 1e-5;
        bool describe = true;
        bool assume_psd = false;

        x_out = solver(A, iA, jA, b, describe, assume_psd, iterations, epsilon);
        for (auto u : x_out)
            output << u << " ";
        output << endl;
    }
    
    cout<<endl<<"Output has been written to output.txt"<<endl;

    return 0;
}
