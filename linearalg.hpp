#include<bits/stdc++.h>
#include<omp.h>
using namespace std;

vector<double> add(vector<double> &a,vector<double> &b, double alpha){  // a+alpha*b
    if(a.size()!=b.size()){                                             // copy of vector c is not made 
        cout<<"Addition of incompatible vectors.\n";                    // due to return value optimization
        exit(0);
    }
    int n=a.size();
    vector<double> c(n);
    #pragma omp parallel for
    for(int i=0;i<n;i++){
        c[i]=a[i]+alpha*b[i];
    }
    return c;
}

double dot(vector<double> &a,vector<double> &b){    // a^T * b
    if(a.size()!=b.size()){
        cout<<"Dot product of incompatible vectors.\n";
        exit(0);
    }
    int n=a.size();
    double sum=0;
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<n;i++){
        sum=sum + (a[i]*b[i]);
    }
    return sum;
}

vector<double> MatVecMult(vector<double> &A,vector<int> &iA,vector<int> &jA, vector<double> &x){ // res=A*x
    int n=x.size();
    vector<double> res(n);
    #pragma omp parallel for
    for(int i=0;i<n;i++){
        res[i]=0;
        #pragma omp parallel for
        for(int idx=iA[i];idx<iA[i+1];idx++){
            res[i]+=A[idx]*x[jA[idx]];
        }
    }
    return res;
}

void vector_copy(vector<double> &in,vector<double> &out){ // out <- in
    if(in.size()!=out.size()){
        cout<<"Copying of incompatible vectors.\n";
        exit(0);
    }
    int n=in.size();
    #pragma omp parallel for
    for(int i=0;i<n;i++){
        out[i]=in[i];
    }
}

void parallel_Conjugate_Gradient(vector<double> &A, vector<int> &iA, vector<int> &jA,vector<double> &b,vector<double> init_x,int iterations){
    
}