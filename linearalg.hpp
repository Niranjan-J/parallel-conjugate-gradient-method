#include<bits/stdc++.h>
#include<omp.h>
using namespace std;

void add(vector<double> &c,vector<double> &a,vector<double> &b, double alpha){  // c=a+alpha*b
    if(a.size()!=b.size()){                                             
        cout<<"Addition of incompatible vectors.\n";                   
        exit(0);
    }
    int n=a.size();
    #pragma omp parallel for
    for(int i=0;i<n;i++){
        c[i]=a[i]+alpha*b[i];
    }
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

void MatVecMult(vector<double> &res,vector<double> &A,vector<int> &iA,vector<int> &jA, vector<double> &x){ // res=A*x
    int n=x.size();
    #pragma omp parallel for
    for(int i=0;i<n;i++){
        res[i]=0;
        #pragma omp parallel for
        for(int idx=iA[i];idx<iA[i+1];idx++){
            res[i]+=A[idx]*x[jA[idx]];
        }
    }
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

//A is n by m
//A_T is m by n
void matrix_transpose(int n, int m, int nz, vector<double> &A, vector<int> &jA, vector<int> &iA,
                      vector<double> &A_T, vector<int> &jA_T, vector<int> &iA_T, bool describe = false)
{
    int i, j, k, l;
    for (i = 0; i <= m; i++)
        iA_T[i] = 0;

    for (i = 0; i < nz; i++)
        iA_T[jA[i] + 1]++;

    for (i = 0; i < m; i++)
        iA_T[i + 1] += iA_T[i];

    auto ptr = iA.begin();

    for (i = 0; i < n; i++, ptr++)
        for (j = *ptr; j < *(ptr + 1); j++)
        {
            k = jA[j];
            l = iA_T[k]++;
            jA_T[l] = i;
            A_T[l] = A[j];
        }

    for (i = m; i > 0; i--)
        iA_T[i] = iA_T[i - 1];

    iA_T[0] = 0;

    if (describe)
    {

        for (auto u : A_T)
        {
            cout << u << " ";
        }
        cout << endl;

        for (auto u : iA_T)
        {
            cout << u << " ";
        }
        cout << endl;

        for (auto u : jA_T)
        {
            cout << u << " ";
        }
        cout << endl;
    }
}

vector<double> parallel_Conjugate_Gradient(vector<double> &A, vector<int> &iA, vector<int> &jA,
vector<double> &b,vector<double> init_x,int iterations){

    // Initialize    
    int n=init_x.size();
    vector<double> matprod(n),x(n),r(n),p(n),rtemp(n);
    vector_copy(init_x,x);
    MatVecMult(matprod,A,iA,jA,x);
    add(r,b,matprod,-1.0);
    vector_copy(r,p);
    int it=0;
    double alpha,beta,r_norm;

    while(it<iterations){
        it++;
        MatVecMult(matprod,A,iA,jA,p);
        r_norm=dot(r,r);
        alpha=r_norm/dot(p,matprod);
        add(x,x,p,alpha);
        add(rtemp,r,matprod,-alpha);
        beta=dot(rtemp,rtemp)/r_norm;
        add(p,rtemp,p,beta);
        vector_copy(rtemp,r);
    }

    return x;
}

