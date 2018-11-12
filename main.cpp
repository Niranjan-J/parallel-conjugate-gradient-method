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

    cout<<endl<<"Testing transpose()"<<endl;
    vector<double> A1{1,3,4,5,6,7,8,9};//n*n
    vector<int> A1row{0,2,5,8};//n+1
    vector<int> A1col{0,2,0,1,2,0,1,2};//nz

    vector<double> A1_T(A1.size());//nz
    vector<int> iA1_T(A1row.size());//n+1
    vector<int> jA1_T(A1col.size());//nz

    matrix_transpose(A1row.size(), A1.size(), A1, A1col, A1row, A1_T,  jA1_T,iA1_T,true);
    return 0;
}