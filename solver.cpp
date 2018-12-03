#include "my_library.hpp"

int main(){

    //Sparse Matrix A (n x n)
    vector<double> A{21,-15,40,-15,75,-20,40,-20,88};
    vector<int> Arow{0,3,6,9};
    vector<int> Acol{0,1,2,0,1,2,0,1,2};
    int n=Arow.size()-1;

    //b (n x 1)
    vector<double> b{16,-260,8};

    //Initial x
    vector<double> x0{1,1,1};

    //Output x
    vector<double> x_out(n);
    
    //Number of iterations and accuracy
    int iterations=10;
    double epsilon=1e-5;
    bool describe=true;
    bool assume_psd=false;

    x_out=solver(A,Arow,Acol,b,describe,assume_psd,iterations,epsilon);

    return 0;
}

/*

Examples for Demonstration:

vector<double> A{21,-15,40,-15,75,-20,40,-20,88};
vector<int> Arow{0,3,6,9};
vector<int> Acol{0,1,2,0,1,2,0,1,2};
vector<double> b{16,-260,8};
Ans: (-4,-4,1);

vector<double> A{2,1,3,2,6,8,6,8,18};
vector<int> Arow{0,3,6,9};
vector<int> Acol{0,1,2,0,1,2,0,1,2};
vector<double> b{1,3,5};
Ans: (0.3,0.4,0);

*/
