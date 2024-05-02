#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <queue>
#include "../include/cost_generator.h"

using namespace std;

const float infi = 100000.0;

/*
Giving user a choice to choose array
Choice 0 : Array is filled with random integers
Choice 1 : 4x4 Test case 1
Choice 2 : 4x4 Test case 2
*/

float *arrInit(Config config)
{
    int SIZE = config.user_n;
    int arrChoice = config.mode;
    float *C;
    if (arrChoice == 0)
    {
        C = generate_cost<float>(config, config.seed);
    }
    else if (arrChoice == 1)
    {
        SIZE = 4;
        float *C = new float[SIZE * SIZE];
        float initC[4][4] = {
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
        float *C = new float[SIZE * SIZE];
        float initC[4][4] = {
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

bool near_zero(float val)
{
    return ((val < eps) && (val > -eps));
}

/*
Choosing the most minimum value in the matrix.
This is used to choose the initial k,l edge (or the most negative dual variable)
*/

float arrminval(float *arr, int SIZE)
{
    float minimum = infi;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (arr[i * SIZE + j] < minimum)
                minimum = arr[i * SIZE + j];

    return minimum;
}

/*
BFS without queue uses an additional phi and change arrays for book-keeping
These 2 arrays are used to reset the values of pred for the paths that are not to be used
*/

bool bfs(int start, int end, bool *visited, float *C, float *u, float *v, int *rows, int *pred, int SIZE)
{
    bool path_found = false;
    int *phi = new int[SIZE];
    for (int i = 0; i < SIZE; i++)
        phi[rows[i]] = i;

    int current_index = 0;
    int next_index = 0;
    int *nodes_to_visit = new int[SIZE];
    nodes_to_visit[next_index++] = start;
    visited[start] = true;

    while (current_index < next_index && path_found == false)
    {
        int node = nodes_to_visit[current_index++];

        for (int j = 0; j < SIZE; j++)
        {
            if (!visited[rows[j]])
            {
                if(near_zero(C[node * SIZE + j] - u[node] - v[j]) == 1)
                {
                    visited[rows[j]] = true;
                    pred[j] = node;
                    nodes_to_visit[next_index++] = rows[j];
                    if (rows[j] == end)
                    {
                        path_found = true;
                        break;
                    }
                }
            }
        }
    }

    bool *change = new bool[SIZE];
    for (int i = 0; i < SIZE; i++)
        change[i] = false;

    if (path_found == true)
    {
        int j = phi[end];
        while (pred[j] != start)
        {
            change[j] = true;
            j = phi[pred[j]];
        }
        change[j] = true;
        for (int i = 0; i < SIZE; i++)
            if (!change[i])
                pred[i] = rows[i];
    }

    if (path_found == false)
    {
        for (int i = 0; i < SIZE; i++)
            pred[i] = rows[i];
    }
    delete[] nodes_to_visit;
    delete[] phi;
    delete[] change;
    return path_found;
}

/*
Identifying the alternating cycle that starts from edge (k,l) which has the most negative dual
The alternating cycle is fulfilled if it returns back to k
*/

bool alternating(int k, int l, bool *SU, bool *SV, bool *LV, int *rows, int *pred, float *C, float *u, float *v, int SIZE)
{
    for (int j = 0; j < SIZE; j++)
        SU[j] = SV[j] = LV[j] = 0;

    LV[l] = 1;

    bool fail = false;
    bool path_found = false;

    int i = rows[l];
    int start = rows[l];
    int end = k;

    SU[start] = 1;

    bool *visited = new bool[SIZE];
    for (int i = 0; i < SIZE; i++)
        visited[i] = 0;

    while (i != k && fail == false)
    {
        SU[i] = 1;
        for (int j = 0; j < SIZE; j++)
            if (LV[j] == 0)
                if(near_zero(C[i * SIZE + j] - u[i] - v[j]) == 1)
                    LV[j] = 1;

        int remNodeCount = 0;
        for (int j = 0; j < SIZE; j++)
            if (LV[j] == 1 && SV[j] == 0)
                remNodeCount++;

        if (remNodeCount == 0)
            fail = true;
        else
        {
            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 1 && SV[j] == 0)
                {
                    SV[j] = 1;
                    i = rows[j];
                    break;
                }
        }
    }

    path_found = bfs(start, end, visited, C, u, v, rows, pred, SIZE);
    // cout<<"Path_found : "<<path_found<<endl;
    if (i == k)
        path_found = true;

    return path_found;
    delete []visited;
}

/*
Finding the least positive slack among visited nodes on left side and unvisited nodes on right side of bipartite graph
*/

float dmin(bool *SU, bool *LV, float *C, float *u, float *v, int SIZE)
{
    // cout<<"slack inside:\n";
    //     for(int i=0; i<SIZE; i++)
    //     {
    //         for(int j=0; j<SIZE; j++)
    //             cout<<C[i*SIZE+j]-u[i]-v[j]<<" ";
    //         cout<<endl;
    //     }

    // cout<<"SUinside:\n";
    // for(int i=0; i<SIZE; i++)
    //     cout<<SU[i]<<" ";
    // cout<<endl;

    // cout<<"LVinside:\n";
    // for(int i=0; i<SIZE; i++)
    //     cout<<LV[i]<<" ";
    // cout<<endl;
    
    float minimum = infi;
    for (int i = 0; i < SIZE; i++)
        if (SU[i] == 1)
            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 0)
                    if (C[i * SIZE + j] - u[i] - v[j] >= eps*100)
                        if (minimum > C[i * SIZE + j] - u[i] - v[j])
                            {
                                minimum = C[i * SIZE + j] - u[i] - v[j];
                                // cout<<"C"<<i<<","<<j<<":"<<C[i * SIZE + j] <<"\tu"<<i<<":"<<u[i]<<"\t-v"<<j<<":"<<v[j]<<"\n";
                                // cout<<"Min : "<<minimum<<endl;
                            }
    // cout<<"Min1 : "<<minimum<<endl;
    return minimum;
}

int* balinski_solve(float *C, int SIZE)
{
    // cout<<"Flag1\n";
    
    float *u = new float[SIZE];         // Dual for row
    float *v = new float[SIZE];         // Dual for column
    int *rows = new int[SIZE];          // To keep track of assigned nodes. Indices:RHS, Values:LHS of Bipartite graph
    int *pred = new int[SIZE];          // To keep track of new assignments
    bool *SU = new bool[SIZE];          // Nodes scanned on LHS of the bipartite graph
    bool *SV = new bool[SIZE];          // Nodes scanned on RHS of the bipartite graph
    bool *LV = new bool[SIZE];          // Book keeping to scan the nodes one by one on RHS
    bool *X = new bool[SIZE * SIZE];      // Assignment matrix
    float *slack = new float[SIZE * SIZE];  // Reduced cost matrix
    
    int *uvrow = new int[SIZE * 3];     // Combined matrix for duals, assignment

    /*
    Initializing row dual = 0 and column dual = diagonal elements of C
    Assign the diagonal in the slack as initial assignment
    */

    // cout<<"Flag2\n";

    for (int i = 0; i < SIZE; i++)
    {
        v[i] = C[i * SIZE + i];
        u[i] = 0;
        rows[i] = pred[i] = i;
        SU[i] = SV[i] = LV[i] = 0;
        for (int j = 0; j < SIZE; j++)
        {
            slack[i * SIZE + j] = C[i * SIZE + j];
            X[i * SIZE + j] = 0;
        }
    }

    
    float checksum = 0;
    for (int i=0; i<SIZE*SIZE; i++)
        checksum += C[i];
    cout<<"Check avg : "<<checksum/(SIZE*SIZE)<<endl;
    

    // cout<<"Flag3\n";

    for (int i = 0; i < SIZE; i++)
    {
        X[rows[i] * SIZE + i] = 1;
        for (int j = 0; j < SIZE; j++)
            slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
    }
    
    float delta = -1;
    float minval = infi;
    int k = -1;
    int l = -1;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (slack[i * SIZE + j] < minval)
            {
                minval = slack[i * SIZE + j];
                k = i;
                l = j;
            }

    bool terminate = false;
    
    // cout<<"C:\n";
    // for(int i=0; i<SIZE; i++)
    // {
    //     for(int j=0; j<SIZE; j++)
    //         cout<<C[i*SIZE+j]<<" ";
    //     cout<<endl;
    // }

    // cout<<"u:\n";
    // for(int i=0; i<SIZE; i++)
    //     cout<<u[i]<<" ";
    // cout<<endl;

    // cout<<"v:\n";
    // for(int i=0; i<SIZE; i++)
    //     cout<<v[i]<<" ";
    // cout<<endl;

    // cout<<"Flag4\n";
    int counter = 0;

    while (arrminval(slack, SIZE) < 0 && terminate == false)
    {
        // cout<<"Flag5\n";
        counter++;
        if(counter>100000)
            break;
        
        // cout<<"slack:\n";
        // for(int i=0; i<SIZE; i++)
        // {
        //     for(int j=0; j<SIZE; j++)
        //         cout<<C[i*SIZE+j]-u[i]-v[j]<<" ";
        //     cout<<endl;
        // }

        // cout<<"u:\n";
        // for(int i=0; i<SIZE; i++)
        //     cout<<u[i]<<" ";
        // cout<<endl;

        // cout<<"v:\n";
        // for(int i=0; i<SIZE; i++)
        //     cout<<v[i]<<" ";
        // cout<<endl;

        // cout<<"SU:\n";
        // for(int i=0; i<SIZE; i++)
        //     cout<<SU[i]<<" ";
        // cout<<endl;

        // cout<<"LV:\n";
        // for(int i=0; i<SIZE; i++)
        //     cout<<LV[i]<<" ";
        // cout<<endl;

        // cout<<"rows:\n";
        // for(int i=0; i<SIZE; i++)
        //     cout<<rows[i]<<" ";
        // cout<<endl;

        bool path_found = alternating(k, l, SU, SV, LV, rows, pred, C, u, v, SIZE);

        // cout<<"SU1:\n";
        // for(int i=0; i<SIZE; i++)
        //     cout<<SU[i]<<" ";
        // cout<<endl;

        // cout<<"LV1:\n";
        // for(int i=0; i<SIZE; i++)
        //     cout<<LV[i]<<" ";
        // cout<<endl;

        // cout<<"rows1:\n";
        // for(int i=0; i<SIZE; i++)
        //     cout<<rows[i]<<" ";
        // cout<<endl;

        // cout<<"Flag5.5\n";
        if (path_found == true)
        {
            // cout<<"Flag6\n";
            for (int i = 0; i < SIZE; i++)
                rows[i] = pred[i];

            rows[l] = k;

            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    X[i * SIZE + j] = 0;
            
            for (int i = 0; i < SIZE; i++)
                X[rows[i] * SIZE + i] = 1;

            delta = u[k] + v[l] - C[k * SIZE + l];
            v[l] -= delta;

            // cout<<"Flag7\n";

            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];

            minval = infi;
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    if (slack[i * SIZE + j] < minval)
                    {
                        minval = slack[i * SIZE + j];
                        k = i;
                        l = j;
                    }

            // cout<<"Flag8\n";
            
            if (slack[k * SIZE + l] >= eps)
                terminate = true;

            // cout<<"Flag9\n";
        }
        else
        {
            // cout<<"Flag10\n";
            // cout<<"Delta1 : "<<delta<<endl;
            delta = dmin(SU, LV, C, u, v, SIZE);
            // cout<<"Delta2 : "<<delta<<endl;
            // cout<<"Flag11\n";
            if (delta == infi)
            {
                
                delta = u[k] + v[l] - C[k * SIZE + l];
                // cout<<"u[k]"<<u[k] <<"\t+ v[l]"<<v[l]<<"\t -C[k,l]"<<C[k * SIZE + l]<<endl;
                // cout<<"Delta1 : "<<delta<<endl;
                // cout<<"k : "<<k<<"\t l : "<<l<<endl;
                // cout<<"slack:\n";
                // for(int i=0; i<SIZE; i++)
                // {
                //     for(int j=0; j<SIZE; j++)
                //         cout<<C[i*SIZE+j]-u[i]-v[j]<<" ";
                //     cout<<endl;
                // }

                // cout<<"u:\n";
                // for(int i=0; i<SIZE; i++)
                //     cout<<u[i]<<" ";
                // cout<<endl;

                // cout<<"v:\n";
                // for(int i=0; i<SIZE; i++)
                //     cout<<v[i]<<" ";
                // cout<<endl;

            }
            
            

            for (int i = 0; i < SIZE; i++)
                if (SU[i] == 1)
                    u[i] += delta;

            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 1)
                    v[j] -= delta;

            if (C[k * SIZE + l] - u[k] - v[l] >= 0)
            {
                // cout<<"Flag CHECK1\n";
                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
            }
            // cout<<"Flag12\n";

            if (slack[k * SIZE + l] >= 0)
            {
                // cout<<"Flag13\n";
                minval = infi;
                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        if (slack[i * SIZE + j] < minval)
                        {
                            minval = slack[i * SIZE + j];
                            k = i;
                            l = j;
                        }
                if (slack[k * SIZE + l] >= 0)
                    terminate = true;
                // cout<<"Flag14\n";
            }
        }        
    }
    // cout<<"Flag15\n";
    float obj = 0.0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            obj += C[i * SIZE + j] * X[i * SIZE + j];
    
    cout<< "Initial Counter : "<<counter<<endl;
    cout << "Initial Objective : " << obj << endl;

    // cout<<"X:\n";
    // for(int i=0; i<SIZE; i++)
    // {
    //     for(int j=0; j<SIZE; j++)
    //         cout<<X[i*SIZE+j]<<" ";
    //     cout<<endl;
    // }
    

    for (int i=0; i<SIZE; i++)
    {
        uvrow[i] = u[i];
        uvrow[i+SIZE]=v[i];
        uvrow[i+2*SIZE]=rows[i];
    }
    // cout<<"Flag16\n";
    delete[] u;
    delete[] v;
    delete[] rows;
    delete[] pred;
    delete[] SU;
    delete[] SV;
    delete[] LV;
    delete[] X;
    delete[] slack;
    // cout<<"Flag17\n";
    return uvrow;
}

int* balinski_resolve(float *C, int SIZE, int *uvrow, float *NC)
{
    
    // cout<<"Flag1\n";
    float *u = new float[SIZE];         // Dual for row
    float *v = new float[SIZE];         // Dual for column
    int *rows = new int[SIZE];          // To keep track of assigned nodes. Indices:RHS, Values:LHS of Bipartite graph
    int *pred = new int[SIZE];          // To keep track of new assignments
    bool *SU = new bool[SIZE];          // Nodes scanned on LHS of the bipartite graph
    bool *SV = new bool[SIZE];          // Nodes scanned on RHS of the bipartite graph
    bool *LV = new bool[SIZE];          // Book keeping to scan the nodes one by one on RHS
    bool *X = new bool[SIZE * SIZE];      // Assignment matrix
    float *slack = new float[SIZE * SIZE];  // Reduced cost matrix
    
    /*
    Adding noise to the new Cost Matrix and preserving original
    */
    float *origC = new float[SIZE * SIZE];
    float *deltaC = new float[SIZE * SIZE];

    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            origC[i*SIZE +j] = C[i*SIZE +j];

    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            C[i*SIZE + j] += NC[i*SIZE + j] * C[i*SIZE + j];
    
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            deltaC[i*SIZE +j] = C[i*SIZE +j] - origC[i*SIZE +j];

    // cout<<"origC:\n";
    // for(int i=0; i<SIZE; i++)
    // {
    //     for(int j=0; j<SIZE; j++)
    //         cout<<origC[i*SIZE+j]<<" ";
    //     cout<<endl;
    // }

    // cout<<"NC:\n";
    // for(int i=0; i<SIZE; i++)
    // {
    //     for(int j=0; j<SIZE; j++)
    //         cout<<NC[i*SIZE+j]<<" ";
    //     cout<<endl;
    // }

    // cout<<"C:\n";
    // for(int i=0; i<SIZE; i++)
    // {
    //     for(int j=0; j<SIZE; j++)
    //         cout<<C[i*SIZE+j]<<" ";
    //     cout<<endl;
    // }

    /*
    Retrieve old duals and assignments
    */
    for (int i = 0; i < SIZE; i++)
    {
        u[i] = uvrow[i];
        v[i] = uvrow[i+SIZE];
        rows[i] = uvrow[i+2*SIZE];
        pred[i] = rows[i];
        SU[i] = SV[i] = LV[i] = 0;
        for (int j = 0; j < SIZE; j++)
        {
            slack[i * SIZE + j] = C[i * SIZE + j];
            X[i * SIZE + j] = 0;
        }
    }

    /*
    Adjust duals to maintain complimentary slackness
    */

    for (int i = 0; i < SIZE; i++)
    {
        u[rows[i]] += deltaC[rows[i]*SIZE+i]/2;
        v[i] += deltaC[rows[i]*SIZE+i]/2;
    }

    /*
    Rest of the code is mostly similar
    */
    for (int i = 0; i < SIZE; i++)
    {
        X[rows[i] * SIZE + i] = 1;
        for (int j = 0; j < SIZE; j++)
            slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
    }
    
    float delta = -1;
    float minval = infi;
    int k = -1;
    int l = -1;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (slack[i * SIZE + j] < minval)
            {
                minval = slack[i * SIZE + j];
                k = i;
                l = j;
            }

    bool terminate = false;

    // cout<<"Flag2\n";
    int counter = 0;
    // cout<<"u:\n";
    // for(int i=0; i<SIZE; i++)
    //     cout<<u[i]<<" ";
    // cout<<endl;

    // cout<<"v:\n";
    // for(int i=0; i<SIZE; i++)
    //     cout<<v[i]<<" ";
    // cout<<endl;

    // cout<<"rows:\n";
    // for(int i=0; i<SIZE; i++)
    //     cout<<rows[i]<<" ";
    // cout<<endl;

    while (arrminval(slack, SIZE) < 0 && terminate == false)
    {
        
        // cout<<"Flag3\n";
        counter++;
        if(counter>50000)
            break;
        bool path_found = alternating(k, l, SU, SV, LV, rows, pred, C, u, v, SIZE);
        // cout<<"Path found : "<<path_found<<endl;
        if (path_found == true)
        {
            // cout<<"Flag4\n";
            for (int i = 0; i < SIZE; i++)
                rows[i] = pred[i];

            rows[l] = k;

            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    X[i * SIZE + j] = 0;
            
            for (int i = 0; i < SIZE; i++)
                X[rows[i] * SIZE + i] = 1;

            delta = u[k] + v[l] - C[k * SIZE + l];
            v[l] -= delta;

            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];

            minval = infi;
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    if (slack[i * SIZE + j] < minval)
                    {
                        minval = slack[i * SIZE + j];
                        k = i;
                        l = j;
                    }

            if (slack[k * SIZE + l] >= eps)
                terminate = true;
            // cout<<"Flag5\n";
        }
        else
        {
            // cout<<"Flag6\n";
            delta = dmin(SU, LV, C, u, v, SIZE);
            // cout<<"delta1:"<<delta<<endl;
            if (delta == infi)
                delta = u[k] + v[l] - C[k * SIZE + l];
            
            // cout<<"delta2:"<<delta<<endl;

            // cout<<"slack:\n";
            // for(int i=0; i<SIZE; i++)
            // {
            //     for(int j=0; j<SIZE; j++)
            //         cout<<C[i*SIZE+j]-u[i]-v[j]<<" ";
            //     cout<<endl;
            // }

            // cout<<"u:\n";
            // for(int i=0; i<SIZE; i++)
            //     cout<<u[i]<<" ";
            // cout<<endl;

            // cout<<"v:\n";
            // for(int i=0; i<SIZE; i++)
            //     cout<<v[i]<<" ";
            // cout<<endl;
            
            
            for (int i = 0; i < SIZE; i++)
                if (SU[i] == 1)
                    u[i] += delta;

            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 1)
                    v[j] -= delta;

            // cout<<"slack1:\n";
            // for(int i=0; i<SIZE; i++)
            // {
            //     for(int j=0; j<SIZE; j++)
            //         cout<<C[i*SIZE+j]-u[i]-v[j]<<" ";
            //     cout<<endl;
            // }

            // cout<<"u1:\n";
            // for(int i=0; i<SIZE; i++)
            //     cout<<u[i]<<" ";
            // cout<<endl;

            // cout<<"v1:\n";
            // for(int i=0; i<SIZE; i++)
            //     cout<<v[i]<<" ";
            // cout<<endl;

            // cout<<"k: "<<k<<"\tl: "<<l<<endl;

            if (C[k * SIZE + l] - u[k] - v[l] >= 0)
                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];

            if (slack[k * SIZE + l] >= 0)
            {
                // cout<<"Flag7\n";
                minval = infi;
                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        if (slack[i * SIZE + j] < minval)
                        {
                            // cout<<"Flag8\n";
                            minval = slack[i * SIZE + j];
                            k = i;
                            l = j;
                        }
                // cout<<"min:"<<minval<<"\tk:"<<k<<"\tl:"<<l<<endl;
                if (slack[k * SIZE + l] >= 0)
                    terminate = true;
            }
        }        
    }

    float obj = 0.0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            obj += C[i * SIZE + j] * X[i * SIZE + j];
    
    cout<< "Counter : "<<counter<<endl;
    cout << "Resolved Objective : " << obj << endl;

    for (int i=0; i<SIZE; i++)
    {
        uvrow[i] = u[i];
        uvrow[i+SIZE]=v[i];
        uvrow[i+2*SIZE]=rows[i];
    }
    // cout<<"X:\n";
    // for(int i=0; i<SIZE; i++)
    // {
    //     for(int j=0; j<SIZE; j++)
    //         cout<<X[i*SIZE+j]<<" ";
    //     cout<<endl;
    // }

    delete[] u;
    delete[] v;
    delete[] rows;
    delete[] pred;
    delete[] SU;
    delete[] SV;
    delete[] LV;
    delete[] X;
    delete[] slack;
    delete[] origC;
    delete[] deltaC;

    return uvrow;
}