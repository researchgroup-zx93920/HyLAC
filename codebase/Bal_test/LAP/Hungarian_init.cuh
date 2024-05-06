#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <queue>
#include "../include/cost_generator.h"

using namespace std;

/*
Giving user a choice to choose array
Choice 0 : Array is filled with random integers
Choice 1 : 4x4 Test case 1
Choice 2 : 4x4 Test case 2
*/

// float *arrInit(Config config)
// {
//     int SIZE = config.user_n;
//     int arrChoice = config.mode;
    
//     if (arrChoice == 0)
//     {
//         float *C;
//         C = generate_cost<float>(config, config.seed);
//         return C;
//     }
//     else if (arrChoice == 1)
//     {
//         SIZE = 4;
//         float *C = new float[SIZE * SIZE];
        
//         float values[SIZE][SIZE] =  {
//                                     {7, 9, 8, 9},
//                                     {2, 8, 5, 7},
//                                     {1, 6, 6, 9},
//                                     {3, 6, 2, 2}
//                                     };

//         for (int i = 0; i < SIZE; ++i)
//             for (int j = 0; j < SIZE; ++j)
//                 C[i * SIZE + j] = values[i][j];

//         for (int i = 0; i < SIZE; ++i)
//         {
//             for (int j = 0; j < SIZE; ++j)
//                 cout<<C[i * SIZE + j] << " ";
//             cout<<endl;
//         }
//         return C;
//     }
//     else if (arrChoice == 2)
//     {
//         SIZE = 4;
//         float *C = new float[SIZE * SIZE];
//         float values[SIZE][SIZE] =  {
//                                     {3, 8, 2, 1},
//                                     {2, 7, 5, 5},
//                                     {9, 8, 1, 2},
//                                     {1, 8, 5, 3}
//                                     };

//         for (int i = 0; i < SIZE; ++i)
//             for (int j = 0; j < SIZE; ++j)
//                 C[i * SIZE + j] = values[i][j];

//         for (int i = 0; i < SIZE; ++i)
//         {
//             for (int j = 0; j < SIZE; ++j)
//                 cout<<C[i * SIZE + j] << " ";
//             cout<<endl;
//         }
        
//         return C;

//     }
//     else
//         cerr << "Invalid choice!" << endl;
//     return 0;
// }

/*
Dealing with floating point equality
*/

bool h_near_zero(float val)
{
    return ((val < eps) && (val > -eps));
}

void rowReduction(float *C, float *u, float *v, int SIZE) 
{
    // Subtract row minima
    for (int i = 0; i < SIZE; ++i)
    {
        float min_val = C[i*SIZE+0];
        for (int j = 1; j < SIZE; ++j)
            if (C[i*SIZE+j] < min_val)
                min_val = C[i*SIZE+j];

        u[i] = min_val;

        for (int j = 0; j < SIZE; ++j)
            C[i*SIZE+j] -= u[i];
    }

    // Subtract column minima
    for (int j = 0; j < SIZE; ++j)
    {
        float min_val = C[0*SIZE+j];
        for (int i = 1; i < SIZE; ++i)
            if (C[i*SIZE+j] < min_val)
                min_val = C[i*SIZE+j];

        v[j] = min_val;

        for (int i = 0; i < SIZE; ++i)
            C[i*SIZE+j] -= v[j];
    }
}

int arrlength(bool *arr, int SIZE)
{
    int length = 0;
    for(int i=0; i<SIZE; i++)
        if(arr[i]==1)
            length++;
    return length;
}

int alternate(int k, float *C, float *u, float *v, int *rows, int *pred, bool *SU, bool *SV, bool *LV, int SIZE)
{
    for(int j=0; j<SIZE; j++)
        SU[j] = SV[j] = LV[j] = 0;

    bool fail = false;
    int sink = -1;
    int i = k;

    while (fail==false && sink==-1)
    {
        
        SU[i]=1;
        for(int j=0; j<SIZE; j++) 
            if (LV[j]==0 && h_near_zero(C[i*SIZE+j]-u[i]-v[j])==1)
            {
                pred[j]=i;
                LV[j]=1;
            }
        
        int remNodeCount = 0;
        for(int j=0; j<SIZE; j++)
            if(LV[j]==1 && SV[j]==0)
                remNodeCount++;

        if (remNodeCount==0)
            fail = true;
        else
        {
            for(int j=0; j<SIZE; j++)
                if(LV[j]==1 && SV[j]==0)
                    {
                        SV[j]=1;
                        if(rows[j]==-1)
                            sink=j;
                        else
                        {
                            i=rows[j];
                            break;
                        }
                    }
        }
    }
    return sink;
}

float h_dmin(bool *SU, bool *LV, float *C, float *u, float *v, int SIZE)
{
    float minimum = 100000.0;
    for(int i=0; i<SIZE; i++)
        if(SU[i]==1)
            for(int j=0; j<SIZE; j++)
                if(LV[j]==0)
                    if(minimum>C[i*SIZE+j] - u[i] - v[j])
                        minimum = C[i*SIZE+j] - u[i] - v[j];
    return minimum;
}

int* hung_seq_solve(float *C, int SIZE)
{
    //srand(time(NULL));
    float *Original = new float[SIZE*SIZE];
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            Original[i*SIZE+j] = C[i*SIZE+j];

    
    int *U = new int[SIZE];
    int *V = new int[SIZE];
    bool *Ubar = new bool[SIZE];
    float *u = new float[SIZE];
    float *v = new float[SIZE];
    int *rows = new int[SIZE];
    int *phi = new int[SIZE];
    int *pred = new int[SIZE];
    bool *SU = new bool[SIZE];
    bool *SV = new bool[SIZE];
    bool *LV = new bool[SIZE];

    int *uvrowh = new int[SIZE * 3];     // Combined matrix for duals, assignment

    for (int i = 0; i < SIZE; ++i)
    {
        Ubar[i] = SU[i] = SV[i] = LV[i] = 0;
        u[i] = v[i] = rows[i] = phi[i] = pred[i] = -1;
        U[i]=V[i]=i;
    }
    
    rowReduction(C,u,v,SIZE);
    float *slack = new float[SIZE*SIZE];
    for(int i=0; i<SIZE; i++)
        for(int j=0; j<SIZE; j++)
        {
            slack[i*SIZE+j] = C[i*SIZE+j];
            C[i*SIZE+j] = Original[i*SIZE+j];
        }

    // Pre-processing
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            if(rows[j]==-1 && h_near_zero(slack[i*SIZE+j])==1)
            {
                rows[j]=i;
                phi[i]=j;
                Ubar[i]=1;
                break;
            }
    int k = -1;
    int counter = 0;
    while(arrlength(Ubar, SIZE) < SIZE)
    {
        counter++;
        if(counter>1000000)
            break;
        
        for(int i=0; i<SIZE; i++)
            if(Ubar[i]==0)
            {
                k = i;
                break;
            }
        while(Ubar[k]==0)
        {
            int sink = alternate(k, C, u, v, rows, pred, SU, SV, LV, SIZE);
            if(sink>-1)
            {
                Ubar[k]=1;
                int b = sink;
                int a = -1;
                while (true)
                {
                    a = pred[b];
                    rows[b] = a;
                    int h = phi[a];
                    phi[a] = b;
                    b = h;
                    if(a==k)
                        break;
                }
            }
            else
            {
                float delta = h_dmin(SU, LV, C, u, v, SIZE);
                for(int i=0; i<SIZE; i++)
                    if(SU[i]==1)
                        u[i]+=delta;
                
                for(int j=0; j<SIZE; j++)
                    if(LV[j]==1)
                        v[j]-=delta;
            }
        }
    }

    bool *X = new bool[SIZE*SIZE];
    for(int i=0; i<SIZE; i++)
        for(int j=0; j<SIZE; j++)
            X[i*SIZE+j]=0;
    
    for(int i=0; i<SIZE; i++)
        X[rows[i]*SIZE+i] = 1;

    int obj = 0;
    for(int i=0; i<SIZE; i++)
        for(int j=0; j<SIZE; j++)
            obj+= C[i*SIZE+j]*X[i*SIZE+j];

    cout<<"Hungarian Counter : "<<counter<<endl;
    cout<<"Hungarian Objective : "<<obj<<endl;

    for (int i=0; i<SIZE; i++)
    {
        uvrowh[i] = u[i];
        uvrowh[i+SIZE]=v[i];
        uvrowh[i+2*SIZE]=rows[i];
    }

    delete[] U;
    delete[] V;
    delete[] Ubar;
    delete[] u;
    delete[] v;
    delete[] rows;
    delete[] phi;
    delete[] pred;
    delete[] SU;
    delete[] SV;
    delete[] LV;
    delete[] slack;
    delete[] Original;

    return uvrowh;

}