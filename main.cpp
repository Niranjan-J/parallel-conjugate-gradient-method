#include "linearalg.hpp"

int main(){
    vector<double> A{21,-15,40,-15,75,-20,40,-20,88};
    vector<int> Arow{0,3,6,9};
    vector<int> Acol{0,1,2,0,1,2,0,1,2};
    vector<double> b{16,-260,8};
    vector<double> x0{1,1,1};
    int iterations=10;
    vector<double> x_out=parallel_Conjugate_Gradient(A,Arow,Acol,b,x0,iterations);
    for(int i=0;i<x_out.size();i++) cout<<x_out[i]<<endl;
    return 0;
}