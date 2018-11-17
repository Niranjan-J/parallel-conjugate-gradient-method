#include "linearalg.hpp"

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

    //Sparse Matrix A_T (n x n)
    vector<double> A_T(A.size());
    vector<int> A_Trow(n+1);
    vector<int> A_Tcol(A.size());
    sparse_matrix_transpose(A,Arow,Acol,A_T,A_Trow,A_Tcol,true);

    clock_t start,end;

    if(!is_symmetric(A,A_T)){
        //R=A^T*A (n x n)
        vector<double> R;
        vector<int> Rrow(n+1);
        vector<int> Rcol;
        A_TA(A_T,A_Trow,A_Tcol,R,Rrow,Rcol,true);

        //q=A^T*b (n x 1)
        vector<double> q(n);
        MatVecMult(q,A_T,A_Trow,A_Tcol,b,true);

        //Solve
        start=clock();
        parallel_Conjugate_Gradient(x_out,R,Rrow,Rcol,q,x0,iterations,epsilon);
        end=clock();
        cout<<"Solution x :\n";
        for(int i=0;i<x_out.size();i++) cout<<x_out[i]<<endl;
        cout<<"\nExecution Time : "<<double(end-start)/CLOCKS_PER_SEC<<"\n\n";        
    }
    else{
        //Solve
        start=clock();
        parallel_Conjugate_Gradient(x_out,A,Arow,Acol,b,x0,iterations,epsilon);
        end=clock();
        cout<<"Solution x :\n";
        for(int i=0;i<x_out.size();i++) cout<<x_out[i]<<endl;        
        cout<<"\nExecution Time : "<<double(end-start)/CLOCKS_PER_SEC<<"\n\n";
    }

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