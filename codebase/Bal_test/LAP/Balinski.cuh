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
    
    if (arrChoice == 0)
    {
        float *C;
        C = generate_cost<float>(config, config.seed);
        return C;
    }
    else if (arrChoice == 1)
    {
        SIZE = 4;
        float *C = new float[SIZE * SIZE];
        
        float values[SIZE][SIZE] =  {
                                    {7, 9, 8, 9},
                                    {2, 8, 5, 7},
                                    {1, 6, 6, 9},
                                    {3, 6, 2, 2}
                                    };

        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                C[i * SIZE + j] = values[i][j];

        for (int i = 0; i < SIZE; ++i)
        {
            for (int j = 0; j < SIZE; ++j)
                cout<<C[i * SIZE + j] << " ";
            cout<<endl;
        }
        return C;
    }
    else if (arrChoice == 2)
    {
        SIZE = 4;
        float *C = new float[SIZE * SIZE];
        float values[SIZE][SIZE] =  {
                                    {3, 8, 2, 1},
                                    {2, 7, 5, 5},
                                    {9, 8, 1, 2},
                                    {1, 8, 5, 3}
                                    };

        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                C[i * SIZE + j] = values[i][j];

        for (int i = 0; i < SIZE; ++i)
        {
            for (int j = 0; j < SIZE; ++j)
                cout<<C[i * SIZE + j] << " ";
            cout<<endl;
        }
        
        return C;

    }
    else
        cerr << "Invalid choice!" << endl;
    return 0;
}

/*
Dealing with floating point equality
*/

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
    int *phi = new int[SIZE];   // Used for backtracking. Index:Row and Value:Column 
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
            if (!visited[rows[j]])
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
    if (i == k)
        path_found = true;

    delete []visited;
    return path_found;
}

/*
Finding the least positive slack among visited nodes on left side and unvisited nodes on right side of bipartite graph
*/

float dmin(bool *SU, bool *LV, float *C, float *u, float *v, int SIZE)
{
    float minimum = infi;
    for (int i = 0; i < SIZE; i++)
        if (SU[i] == 1)
            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 0)
                    if (C[i * SIZE + j] - u[i] - v[j] >= eps)
                        if (minimum > C[i * SIZE + j] - u[i] - v[j])
                            minimum = C[i * SIZE + j] - u[i] - v[j];

    return minimum;
}

void initializeMemory   (float* &u, float* &v, 
                        int* &rows, int* &pred, 
                        bool* &SU, bool* &SV, bool* &LV, bool* &X, 
                        float* &slack, int SIZE) 
{
    u = new float[SIZE];         // Dual for row
    v = new float[SIZE];         // Dual for column
    rows = new int[SIZE];        // To keep track of assigned nodes. Indices: RHS, Values: LHS of Bipartite graph
    pred = new int[SIZE];        // To keep track of new assignments
    SU = new bool[SIZE];         // Nodes scanned on LHS of the bipartite graph
    SV = new bool[SIZE];         // Nodes scanned on RHS of the bipartite graph
    LV = new bool[SIZE];         // Bookkeeping to scan the nodes one by one on RHS
    X = new bool[SIZE * SIZE];   // Assignment matrix
    slack = new float[SIZE * SIZE];  // Reduced cost matrix
}

void cleanupMemory  (float* &u, float* &v, 
                    int* &rows, int* &pred, 
                    bool* &SU, bool* &SV, bool* &LV, bool* &X, 
                    float* &slack)
{
    delete[] u;
    delete[] v;
    delete[] rows;
    delete[] pred;
    delete[] SU;
    delete[] SV;
    delete[] LV;
    delete[] X;
    delete[] slack;
}

int* bal_common(
    float *C, float *u, float *v, int SIZE, float *slack, 
    bool *SU, bool *SV, bool *LV,
    bool *X, int *rows, int *pred)
{
    int *uvrowc = new int[SIZE * 3];
    for (int i=0; i<SIZE; i++)
        SU[i] = SV[i] = LV[i] = 0;

    float checksum = 0;
    for (int i=0; i<SIZE*SIZE; i++)
        checksum += C[i];
    cout<<"Check avg : "<<checksum/(SIZE*SIZE)<<endl;

    float delta = -1;
    float minval = infi;
    int k = -1;
    int l = -1;

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (slack[i * SIZE + j] - minval < eps)
            {
                minval = slack[i * SIZE + j];
                k = i;
                l = j;
            }

    bool terminate = false;

    int counter = 0;

    while (arrminval(slack, SIZE) < 0 && terminate == false)
    {
        counter++;
        if(counter>1000000)
            break;

        bool path_found = alternating(k, l, SU, SV, LV, rows, pred, C, u, v, SIZE);

        if (path_found == true)
        {
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
                    if (slack[i * SIZE + j] -minval < eps)
                    {
                        minval = slack[i * SIZE + j];
                        k = i;
                        l = j;
                    }

            if (slack[k * SIZE + l] >= eps)
                terminate = true;
        }
        else
        {
            delta = dmin(SU, LV, C, u, v, SIZE);
            if (delta == infi)
            {
                delta = u[k] + v[l] - C[k * SIZE + l];
            }

            for (int i = 0; i < SIZE; i++)
                if (SU[i] == 1)
                    u[i] += delta;

            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 1)
                    v[j] -= delta;

            if (C[k * SIZE + l] - u[k] - v[l] >= eps)
            {

                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
            }

            if (slack[k * SIZE + l] >= eps)
            {

                minval = infi;
                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        if (slack[i * SIZE + j] - minval < eps)
                        {
                            minval = slack[i * SIZE + j];
                            k = i;
                            l = j;
                        }
                if (slack[k * SIZE + l] >= eps)
                    terminate = true;

            }
        }        
    }

    float obj = 0.0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            obj += C[i * SIZE + j] * X[i * SIZE + j];

    cout<< "Balinski Counter : "<<counter<<endl;
    cout << "Balinski Objective : " << obj << endl;

    for (int i=0; i<SIZE; i++)
    {
        uvrowc[i] = u[i];
        uvrowc[i+SIZE]=v[i];
        uvrowc[i+2*SIZE]=rows[i];
    }

    return uvrowc;
}

int* balinski_solve(float *C, int SIZE)
{
    float *u, *v;
    int *rows, *pred;
    bool *SU, *SV, *LV, *X;
    float *slack;
    initializeMemory    (u, v, rows, pred, 
                        SU, SV, LV, X, slack, SIZE);

    
    int *uvrow = new int[SIZE * 3];     // Combined matrix for duals, assignment

    /*
    Initializing row dual = 0 and column dual = diagonal elements of C
    Assign the diagonal in the slack as initial assignment
    */

    for (int i = 0; i < SIZE; i++)
    {
        v[i] = C[i * SIZE + i];
        u[i] = 0;
        rows[i] = pred[i] = i;
        for (int j = 0; j < SIZE; j++)
        {
            slack[i * SIZE + j] = C[i * SIZE + j];
            X[i * SIZE + j] = 0;
        }
    }

    for (int i = 0; i < SIZE; i++)
    {
        X[rows[i] * SIZE + i] = 1;
        for (int j = 0; j < SIZE; j++)
            slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
    }

    uvrow = bal_common  (C, u, v, SIZE, slack, 
                        SU, SV, LV, 
                        X, rows, pred);

    cleanupMemory(u, v, rows, pred, SU, SV, LV, X, slack);

    return uvrow;

}

int* balinski_resolve(float *C, int SIZE, int *uvrow, float *NC, int precision)
{
    float *u, *v;
    int *rows, *pred;
    bool *SU, *SV, *LV, *X;
    float *slack;

    initializeMemory    (u, v, rows, pred, 
                        SU, SV, LV, X, slack, SIZE);

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
            C[i*SIZE + j] += roundf(NC[i*SIZE + j] * C[i*SIZE + j]*pow(10,precision))/pow(10,precision);
    
    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            deltaC[i*SIZE +j] = C[i*SIZE +j] - origC[i*SIZE +j];

    /*
    Retrieve old duals and assignments
    */
    for (int i = 0; i < SIZE; i++)
    {
        u[i] = uvrow[i];
        v[i] = uvrow[i+SIZE];
        rows[i] = uvrow[i+2*SIZE];
        pred[i] = rows[i];
        for (int j = 0; j < SIZE; j++)
        {
            slack[i * SIZE + j] = C[i * SIZE + j];
            X[i * SIZE + j] = 0;
        }
    }

    for (int i = 0; i < SIZE; i++)
    {
        u[rows[i]] += ceil(deltaC[rows[i]*SIZE+i]/2*pow(10,precision))/pow(10,precision);
        v[i] += floor(deltaC[rows[i]*SIZE+i]/2*pow(10,precision))/pow(10,precision);
    }

    for (int i = 0; i < SIZE; i++)
    {
        X[rows[i] * SIZE + i] = 1;
        for (int j = 0; j < SIZE; j++)
            slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
    }

    uvrow = bal_common  (C, u, v, SIZE, slack, 
                        SU, SV, LV, 
                        X, rows, pred);

    cleanupMemory(u, v, rows, pred, SU, SV, LV, X, slack);

    return uvrow;

}