# include <iostream>
# include <cstdlib>
# include <ctime>
# include <fstream>
#include "../include/cost_generator.h"

using namespace std;

// const int SIZE = 40;

// void arrInit(int arrChoice, int *C, int SIZE) {
//     if (arrChoice == 0) {
//         // Fill with random integers ranging from 0 to 10
//         for (int i = 0; i < SIZE; ++i) {
//             for (int j = 0; j < SIZE; ++j) {
//                 C[i*SIZE+j] = rand() % 15 + rand()%12 + rand()%7;
//             }
//         }
//     } else if (arrChoice == 1) {
//         int initC[SIZE][SIZE] = {
//             {7, 9, 8, 9},
//             {2, 8, 5, 7},
//             {1, 6, 6, 9},
//             {3, 6, 2, 2}
//         };
//         for (int i = 0; i < SIZE; ++i) {
//             for (int j = 0; j < SIZE; ++j) {
//                 C[i*SIZE+j] = initC[i][j];
//             }
//         }
//     } else if (arrChoice == 2) {
//         int initC[SIZE][SIZE] = {
//             {3, 8, 2, 1},
//             {2, 7, 5, 5},
//             {9, 8, 1, 2},
//             {1, 8, 5, 3}
//         };
//         for (int i = 0; i < SIZE; ++i) {
//             for (int j = 0; j < SIZE; ++j) {
//                 C[i*SIZE+j] = initC[i][j];
//             }
//         }
//     } else {
//         cerr << "Invalid choice!" << endl;
//     }
// }

int *arrInit(Config config)
{
    int SIZE = config.user_n;
    int arrChoice = config.mode;
    int *C;
    if (arrChoice == 0)
    {
        C = generate_cost<int>(config, config.seed);
    }
    else if (arrChoice == 1)
    {
        SIZE = 4;
        int *C = new int[SIZE * SIZE];
        int initC[4][4] = {
            {7, 9, 8, 9},
            {2, 8, 5, 7},
            {1, 6, 6, 9},
            {3, 6, 2, 2}};
        for (int i = 0; i < SIZE; ++i)
        {
            for (int j = 0; j < SIZE; ++j)
            {
                C[i * SIZE + j] = initC[i][j];
            }
        }
    }
    else if (arrChoice == 2)
    {
        SIZE = 4;
        int *C = new int[SIZE * SIZE];
        int initC[4][4] = {
            {3, 8, 2, 1},
            {2, 7, 5, 5},
            {9, 8, 1, 2},
            {1, 8, 5, 3}};
        for (int i = 0; i < SIZE; ++i)
        {
            for (int j = 0; j < SIZE; ++j)
            {
                C[i * SIZE + j] = initC[i][j];
            }
        }
    }
    else
    {
        cerr << "Invalid choice!" << endl;
    }
    return C;
}

void print2DArray(int *arr, int SIZE) {
    cout << endl;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            cout << arr[i*SIZE+j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printfile2DArray(int *arr, ofstream& outputFile, int SIZE) {
    outputFile << endl;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            outputFile << arr[i*SIZE+j];
            if (j < SIZE - 1)
                outputFile << ",";
        }
        outputFile << endl;
    }
    outputFile << endl;
}

void print1DArray(int *arr, int SIZE) {
    for (int i = 0; i < SIZE; ++i)
            cout << arr[i] << " ";
    cout << endl;
}



void rowReduction(int *C, int *u, int *v, int SIZE) {
    // Subtract row minima
    for (int i = 0; i < SIZE; ++i) {
        int min_val = C[i*SIZE+0];
        for (int j = 1; j < SIZE; ++j) {
            if (C[i*SIZE+j] < min_val) {
                min_val = C[i*SIZE+j];
            }
        }
        u[i] = min_val;

        for (int j = 0; j < SIZE; ++j) {
            C[i*SIZE+j] -= u[i];
        }
    }

    // Subtract column minima
    for (int j = 0; j < SIZE; ++j) {
        int min_val = C[0*SIZE+j];
        for (int i = 1; i < SIZE; ++i) {
            if (C[i*SIZE+j] < min_val) {
                min_val = C[i*SIZE+j];
            }
        }
        v[j] = min_val;

        for (int i = 0; i < SIZE; ++i) {
            C[i*SIZE+j] -= v[j];
        }
    }
}

int arrlength(int *arr, int SIZE)
{
    int length = 0;
    for(int i=0; i<SIZE; i++)
        if(arr[i]==1)
            length++;
    return length;
}

int alternate(int k, int *C, int *u, int *v, int *rows, int *pred, int *SU, int *SV, int *LV, int SIZE)
{
    /* cout<<"*****Entered Alternating****"<<endl;
    cout<<"k : "<<k<<endl<<"Entering row : ";
    print1DArray(rows);
    cout<<"Entering pred : ";
    print1DArray(pred);
    cout<<"Entering u : ";
    print1DArray(u);
    cout<<"Entering v : ";
    print1DArray(v); */
    for(int j=0; j<SIZE; j++)
    {
        SU[j] = SV[j] = LV[j] = -1;
    }
    // cout<<"SUinit:\n";
    // print1DArray(SU);
    // cout<<"SVinit:\n";
    // print1DArray(SV);
    // cout<<"LVinit:\n";
    // print1DArray(LV);
    

    bool fail = false;
    int sink = -1;
    int i = k;

    while (fail==false && sink==-1)
    {
        
        SU[i]=1;
        for(int j=0; j<SIZE; j++) 
            if (LV[j]==-1 && C[i*SIZE+j]-u[i]-v[j]==0)
            {
                // cout<<"LV["<<j<<"]:"<<LV[j]<<endl;
                // cout<<"C["<<i<<"]["<<j<<"]:"<<C[i][j]<<endl;
                // cout<<"u["<<i<<"]:"<<u[i]<<endl;
                // cout<<"v["<<j<<"]:"<<v[j]<<endl;
                pred[j]=i;
                LV[j]=1;
            }

        // cout<<"LV CHECK:\n";
        // print1DArray(LV);
        // cout<<"SV CHECK:\n";
        // print1DArray(SV);

        int remNodeCount = 0;
        for(int j=0; j<SIZE; j++)
            if(LV[j]==1 && SV[j]==-1)
                remNodeCount++;

        if (remNodeCount==0)
            fail = true;
        else
        {
            for(int j=0; j<SIZE; j++)
                if(LV[j]==1 && SV[j]==-1)
                    {
                        SV[j]=1;
                        if(rows[j]==-1)
                            sink=j;
                        else
                        {
                            i=rows[j];
                            // cout<<"i : "<<i<<endl;
                            break;
                        }
                    }
        }
        // cout<<"SU after processing:\n";
        // print1DArray(SU);
        // cout<<"SV after processing:\n";
        // print1DArray(SV);
        // cout<<"LV after processing:\n";
        // print1DArray(LV);
        
    }
    // cout<<"Rows inside func: ";
    // print1DArray(rows);
    // cout<<"Pred inside func: ";
    // print1DArray(pred);
    return sink;

}

int dmin(int *SU, int *LV, int *C, int *u, int *v, int SIZE)
{
    // cout<<"SU in dmin:\n";
    // print1DArray(SU);
    int minimum = 100000;
    for(int i=0; i<SIZE; i++)
        if(SU[i]==1)
            for(int j=0; j<SIZE; j++)
                if(LV[j]==-1)
                    if(minimum>C[i*SIZE+j] - u[i] - v[j])
                        minimum = C[i*SIZE+j] - u[i] - v[j];
    return minimum;
}

int hung_seq_solve(int *C, int SIZE)
{
    
    srand(time(NULL));
    // int arrChoice;
    // cout << "Enter choice (0 (to randomize), 1, or 2): ";
    // cin >> arrChoice;

    // ofstream outputFile("HungOutput.csv");
    // if (!outputFile.is_open()) {
    //     cerr << "Error opening output.csv" << endl;
    //     return 1;
    // }


    // int *C = new int[SIZE*SIZE];
    // arrInit(arrChoice, C, SIZE);

    // cout<<"Cost matrix C:\n";
    // print2DArray(C);

    int *Original = new int[SIZE*SIZE];
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            Original[i*SIZE+j] = C[i*SIZE+j];

    
    int *U = new int[SIZE];
    int *V = new int[SIZE];
    int *Ubar = new int[SIZE];
    int *u = new int[SIZE];
    int *v = new int[SIZE];
    int *rows = new int[SIZE];
    int *phi = new int[SIZE];
    int *pred = new int[SIZE];
    int *SU = new int[SIZE];
    int *SV = new int[SIZE];
    int *LV = new int[SIZE];


    for (int i = 0; i < SIZE; ++i)
    {
        Ubar[i] = u[i] = v[i] = rows[i] = phi[i] = pred[i] = SU[i] = SV[i] = LV[i] = -1;
        U[i]=V[i]=i;
    }
    
    rowReduction(C,u,v,SIZE);
    int *slack = new int[SIZE*SIZE];
    for(int i=0; i<SIZE; i++)
        for(int j=0; j<SIZE; j++)
        {
            slack[i*SIZE+j] = C[i*SIZE+j];
            C[i*SIZE+j] = Original[i*SIZE+j];
        }

    // Pre-processing
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            if(rows[j]==-1 && slack[i*SIZE+j]==0)
            {
                rows[j]=i;
                phi[i]=j;
                Ubar[i]=1;
                break;
            }

    // cout<<"Reduced Cost matrix Slack:\n";
    // print2DArray(slack, SIZE);
    // cout<<"u:\n";
    // print1DArray(u, SIZE);
    // cout<<"v:\n";
    // print1DArray(v, SIZE);
    // cout<<"rows:\n";
    // print1DArray(rows);
    // cout<<"phi:\n";
    // print1DArray(phi);
    // cout<<"Ubar:\n";
    // print1DArray(Ubar);
    
    // int dum_k = 1;
    // int sink = alternate(dum_k, C, V, u, v, rows, pred, SU, SV, LV);
    // cout<<"Sink :"<<sink<<endl<<endl;

    // cout<<"SU:\n";
    // print1DArray(SU);
    // cout<<"SV:\n";
    // print1DArray(SV);
    // cout<<"LV:\n";
    // print1DArray(LV);

    // int j=sink;
    // int i=-1;
    // while (true)
    // {
    //     i = pred[j];
    //     rows[j] = i;
    //     int h = phi[i];
    //     phi[i] = j;
    //     j = h;
    //     if(i==dum_k)
    //         break;
    // }

    // cout<<"rows:\n";
    // print1DArray(rows);
    // cout<<"pred:\n";
    // print1DArray(pred);



    // int dum_k = 1;
    // int sink = alternate(dum_k, C, V, u, v, rows, pred, SU, SV, LV);
    // cout<<"Sink :"<<sink<<endl<<endl;

    // cout<<"SU:\n";
    // print1DArray(SU);
    // cout<<"SV:\n";
    // print1DArray(SV);
    // cout<<"LV:\n";
    // print1DArray(LV);



    int k = -1;
    //int Ubarlength = 0;

    while(arrlength(Ubar, SIZE) < SIZE)
    {
        // counter1++;
        // cout<<"Counter1 : "<<counter1<<endl;
        for(int i=0; i<SIZE; i++)
            if(Ubar[i]==-1)
            {
                k = i;
                break;
            }

        // cout<<"u:\n";
        // print1DArray(u);
        // cout<<"v:\n";
        // print1DArray(v);

        while(Ubar[k]==-1)
        {
            // counter2++;
            // cout<<"Counter2 : "<<counter2<<endl;

            // cout<<"u:\n";
            // print1DArray(u);
            // cout<<"v:\n";
            // print1DArray(v);
            // cout<<"Main k :"<<k<<endl;
            int sink = alternate(k, C, u, v, rows, pred, SU, SV, LV, SIZE);
            
            // cout<<"Main Sink : "<<sink<<endl;
            // cout<<"SU:\n";
            // print1DArray(SU);
            // cout<<"SV:\n";
            // print1DArray(SV);
            // cout<<"rows:\n";
            // print1DArray(rows);

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
                    {
                        // cout<<"Inside rows:\n";
                        // print1DArray(rows);
                        break;
                    }

                }

            }
            else
            {
                // cout<<"Entered"<<endl;
                int delta = dmin(SU, LV, C, u, v, SIZE);
                // cout<<"Delta :"<<delta<<endl;
                for(int i=0; i<SIZE; i++)
                    if(SU[i]==1)
                        u[i]+=delta;
                
                for(int j=0; j<SIZE; j++)
                    if(LV[j]==1)
                        v[j]-=delta;
            }
        }
    }

    int *X = new int[SIZE*SIZE];
    for(int i=0; i<SIZE; i++)
        for(int j=0; j<SIZE; j++)
            X[i*SIZE+j]=0;
    
    for(int i=0; i<SIZE; i++)
        X[rows[i]*SIZE+i] = 1;

    // cout<<"Assignment matrix X:\n";
    // print2DArray(X);

    int obj = 0;
    for(int i=0; i<SIZE; i++)
        for(int j=0; j<SIZE; j++)
            obj+= C[i*SIZE+j]*X[i*SIZE+j];

    // cout<<"Cost matrix C:\n";
    // print2DArray(C);
    cout<<"Obj : "<<obj<<endl;

    // outputFile << "Cost matrix C:\n";
    // printfile2DArray(C, outputFile, SIZE);

    // outputFile << "Assignment matrix X:\n";
    // printfile2DArray(X, outputFile, SIZE);

    // outputFile << "Obj : " << obj << endl;

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

    return 0;
}