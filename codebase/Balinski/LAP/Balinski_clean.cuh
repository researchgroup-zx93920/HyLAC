#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <queue>
#include "../include/cost_generator.h"

using namespace std;

const int infi = 100000;

/*
Giving user a choice to choose array
Choice 0 : Array is filled with random integers
Choice 1 : 4x4 Test case 1
Choice 2 : 4x4 Test case 2
*/

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

/*
Functions to print the matrices
*/

void print2DArray(int *arr, int SIZE)
{
    cout << endl;
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            cout << arr[i * SIZE + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void print1DArray(int *arr, int SIZE)
{
    for (int i = 0; i < SIZE; ++i)
        cout << arr[i] << " ";
    cout << endl;
}

void print1DBoolArray(bool *arr, int SIZE)
{
    for (int i = 0; i < SIZE; ++i)
        cout << arr[i] << " ";
    cout << endl;
}

/*
Functions to export the matrices to a csv file
*/

void printfile2DArray(int *arr, ofstream &outputFile, int SIZE)
{
    outputFile << endl;
    for (int i = 0; i < SIZE; ++i)
    {
        for (int j = 0; j < SIZE; ++j)
        {
            outputFile << arr[i * SIZE + j];
            if (j < SIZE - 1)
                outputFile << ",";
        }
        outputFile << endl;
    }
    outputFile << endl;
}

/*
Choosing the most minimum value in the matrix.
This is used to choose the initial k,l edge (or the most negative dual variable)
*/

int arrminval(int *arr, int SIZE)
{
    int minimum = infi;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (arr[i * SIZE + j] < minimum)
                minimum = arr[i * SIZE + j];

    return minimum;
}

/*
Using DFS to find an alternating tree that will lead us back to the starting node k.
If (k,l) is the starting edge going from the left side of the bipartite graph to the
right side, for the dfs we consider the starting node to be row[l], which is essentially
the pre-existing assignment of node l from the right side of the bipartite graph to the left side.
Here node = rows[l] (starting node) and end = k (ending node)
rows indicates the existing assignments
pred is where we store the new assignments
*/

bool dfs(int node, int end, bool *visited, int *C, int *u, int *v, int *rows, int *pred, int SIZE)
{
    if (node == end)
        return true;
    for (int j = 0; j < SIZE; j++)
    {
        if (visited[j] == 0 && C[node * SIZE + j] - u[node] - v[j] == 0)
        {
            visited[j] = 1;
            if (dfs(rows[j], end, visited, C, u, v, rows, pred, SIZE) == true)
            {
                pred[j] = node; // Adding nodes to pred, as path forms
                return true;
            }
        }
        pred[j] = rows[j]; // If path is not found, pred reverts back to pre-existing assignments (rows array)
    }
    return false;
}

/*
BFS uses an additional phi and change arrays for book-keeping
These 2 arrays are used to reset the values of pred for the paths that are not to be used
*/

bool bfs(int start, int end, bool *visited, int *C, int *u, int *v, int *rows, int *pred, int SIZE)
{
    queue<int> q;
    q.push(start);
    visited[start] = true;
    bool path_found = false;
    int *phi = new int[SIZE];
    // phi keeps track of the already assigned nodes in the reverse order as 'rows' matrix. Indices indicate left side of the bipartite graph and values the right side
    for (int i = 0; i < SIZE; i++)
        phi[rows[i]] = i;

    while (!q.empty() && path_found == false)
    {
        int node = q.front();
        q.pop();

        for (int j = 0; j < SIZE; j++)
        {
            if (!visited[rows[j]] && C[node * SIZE + j] - u[node] - v[j] == 0)
            {
                visited[rows[j]] = true;
                pred[j] = node;
                q.push(rows[j]);
                if (rows[j] == end)
                {
                    path_found = true;
                    break;
                }
            }
        }
    }
    bool *change = new bool[SIZE]; // Book-keeping. We want to backtrack from the final node to the original node and make sure to change only the nodes on the required path
    for (int i = 0; i < SIZE; i++)
        change[i] = 0;

    if (path_found == true)
    {
        int j = phi[end];
        // Back-tracking
        while (pred[j] != start)
        {
            change[j] = 1;
            j = phi[pred[j]];
        }
        change[j] = 1;
        // Only changing the required variables
        for (int i = 0; i < SIZE; i++)
            if (change[i] == 0)
                pred[i] = rows[i];
    }
    // If path is not found, reset all pred to rows
    if (path_found == false)
    {
        for (int i = 0; i < SIZE; i++)
            pred[i] = rows[i];
    }
    return path_found;
}

/*
BFS without queue
*/

bool bfs2(int start, int end, bool *visited, int *C, int *u, int *v, int *rows, int *pred, int SIZE)
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
            if (!visited[rows[j]] && C[node * SIZE + j] - u[node] - v[j] == 0)
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

bool alternating(int k, int l, bool *SU, bool *SV, bool *LV, int *rows, int *pred, int *C, int *u, int *v, int SIZE)
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
            if (LV[j] == 0 && C[i * SIZE + j] - u[i] - v[j] == 0)
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

    /*
    Use BFS or DFS by commenting out whichever is not needed. Use BFS2 to avoid queue
    */

    // path_found = bfs(start, end, visited, C, u, v, rows, pred, SIZE);
    path_found = bfs2(start, end, visited, C, u, v, rows, pred, SIZE);
    // path_found = dfs(start, end, visited, C, u, v, rows, pred, SIZE);

    if (i == k)
        path_found = true;

    return path_found;
}

/*
Finding the least positive slack among visited nodes on left side and unvisited nodes on right side of bipartite graph
*/

int dmin(bool *SU, bool *LV, int *C, int *u, int *v, int SIZE)
{
    int minimum = infi;
    for (int i = 0; i < SIZE; i++)
        if (SU[i] == 1)
            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 0)
                    if (C[i * SIZE + j] - u[i] - v[j] >= 0)
                        if (minimum > C[i * SIZE + j] - u[i] - v[j])
                            minimum = C[i * SIZE + j] - u[i] - v[j];
    return minimum;
}

// int main(int *hcost, int n_size)
int* balinski_solve(int *C, int SIZE)
{
    // Remove commenting to print the cost matrix
    /*
        cout<<"Cost matrix C:\n";
        print2DArray(C, SIZE);
    */

    // ofstream outputFile("BalinskiOutput.csv");
    // if (!outputFile.is_open())
    // {
    //     cerr << "Error opening output.csv" << endl;
    //     return 1;
    // }

    int *u = new int[SIZE];        // Dual for row
    int *v = new int[SIZE];        // Dual for column
    int *rows = new int[SIZE];     // To keep track of assigned nodes. Indices:RHS, Values:LHS of Bipartite graph
    int *pred = new int[SIZE];     // To keep track of new assignments
    bool *SU = new bool[SIZE];     // Nodes scanned on LHS of the bipartite graph
    bool *SV = new bool[SIZE];     // Nodes scanned on RHS of the bipartite graph
    bool *LV = new bool[SIZE];     // Book keeping to scan the nodes one by one on RHS
    int *X = new int[SIZE * SIZE]; // Assignment matrix
    int *slack = new int[SIZE * SIZE];
    int *uvrowpred = new int[SIZE * 4];

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
    for (int i = 0; i < SIZE; i++)
    {
        X[rows[i] * SIZE + i] = 1;
        for (int j = 0; j < SIZE; j++)
            slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
    }
    int delta = -1;

    // Remove commenting to print the arrays

    /*
        cout<<"u:\n";
        print1DArray(u, SIZE);
        cout<<"v:\n";
        print1DArray(v, SIZE);
        cout<<"rows:\n";
        print1DArray(rows, SIZE);
        cout<<"phi:\n";
        print1DArray(phi, SIZE);
        cout<<"pred:\n";
        print1DArray(pred, SIZE);
        cout<<"SU:\n";
        print1DBoolArray(SU, SIZE);
        cout<<"SV:\n";
        print1DBoolArray(SV, SIZE);
        cout<<"LV:\n";
        print1DBoolArray(LV, SIZE);
        cout<<"X:\n";
        print2DArray(X, SIZE);
        cout<<"slack:\n";
        print2DArray(slack, SIZE);

    */

    int minval = infi;
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

    // Remove commenting to print the values of k and l
    /*   cout<<"k : "<<k<<endl<<"l : "<<l<<endl; */

    while (arrminval(slack, SIZE) < 0 && terminate == false)
    {
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
                    if (slack[i * SIZE + j] < minval)
                    {
                        minval = slack[i * SIZE + j];
                        k = i;
                        l = j;
                    }

            if (slack[k * SIZE + l] >= 0)
                terminate = true;
        }
        else
        {
            delta = dmin(SU, LV, C, u, v, SIZE);
            if (delta == infi)
                delta = u[k] + v[l] - C[k * SIZE + l];

            for (int i = 0; i < SIZE; i++)
                if (SU[i] == 1)
                    u[i] += delta;

            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 1)
                    v[j] -= delta;

            if (u[k] + v[l] <= C[k * SIZE + l])
                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];

            if (slack[k * SIZE + l] >= 0)
            {
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
            }
        }
    }

    // Remove comments to console print
    /*
        cout<<"Cost matrix C:\n";
        print2DArray(C, SIZE);

        cout<<"Assignment matrix X:\n";
        print2DArray(X, SIZE);

    */

    // outputFile << "Cost matrix C:\n";
    // printfile2DArray(C, outputFile, SIZE);

    // outputFile << "Assignment matrix X:\n";
    // printfile2DArray(X, outputFile, SIZE);

    int obj = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            obj += C[i * SIZE + j] * X[i * SIZE + j];

    cout << "Obj : " << obj << endl;
    // outputFile << "Obj : " << obj << endl;

    for (int i=0; i<SIZE; i++)
    {
        uvrowpred[i] = u[i];
        uvrowpred[i+SIZE]=v[i];
        uvrowpred[i+2*SIZE]=rows[i];
        uvrowpred[i+3*SIZE]=pred[i];
    }

    delete[] u;
    delete[] v;
    delete[] rows;
    delete[] pred;
    delete[] SU;
    delete[] SV;
    delete[] LV;
    delete[] X;
    delete[] slack;

    return uvrowpred;
}

int* balinski_resolve(int *C, int SIZE, int *uvrowpred, float noise)
{
    // Remove commenting to print the cost matrix
    /*
        cout<<"Cost matrix C:\n";
        print2DArray(C, SIZE);
    */

    // ofstream outputFile("BalinskiOutput.csv");
    // if (!outputFile.is_open())
    // {
    //     cerr << "Error opening output.csv" << endl;
    //     return 1;
    // }

    int *origC = new int[SIZE * SIZE];
    int *deltaC = new int[SIZE * SIZE];
    for (int i=0; i<SIZE; i++)
    {
        for (int j=0; j<SIZE; j++)
        {
            origC[i*SIZE +j] = C[i*SIZE +j];
            // cout<<origC[i*SIZE +j]<<" ";
        }
        // cout<<endl;
    }

    // cout<<endl<<endl;
    for (int i = 0; i < SIZE; i++)
    {
        C[i] = C[i] + static_cast<int>(C[i] * noise);
    }

    // for (int i=0; i<SIZE; i++)
    // {
    //     for (int j=0; j<SIZE; j++)
    //     {
    //         C[i*SIZE +j] = C[i*SIZE +j];
    //         cout<<C[i*SIZE +j]<<" ";
    //     }
    //     cout<<endl;
    // }


    for (int i=0; i<SIZE; i++)
        for (int j=0; j<SIZE; j++)
            deltaC[i*SIZE +j] = C[i*SIZE +j] - origC[i*SIZE +j];


    int *u = new int[SIZE];        // Dual for row
    int *v = new int[SIZE];        // Dual for column
    int *rows = new int[SIZE];     // To keep track of assigned nodes. Indices:RHS, Values:LHS of Bipartite graph
    int *pred = new int[SIZE];     // To keep track of new assignments
    bool *SU = new bool[SIZE];     // Nodes scanned on LHS of the bipartite graph
    bool *SV = new bool[SIZE];     // Nodes scanned on RHS of the bipartite graph
    bool *LV = new bool[SIZE];     // Book keeping to scan the nodes one by one on RHS
    int *X = new int[SIZE * SIZE]; // Assignment matrix
    int *slack = new int[SIZE * SIZE];

    for (int i = 0; i < SIZE; i++)
    {
        v[i] = uvrowpred[i+SIZE];
        u[i] = uvrowpred[i];
        rows[i] = uvrowpred[i+2*SIZE];
        pred[i] = uvrowpred[i+3*SIZE];
        // rows[i] = pred[i] = i;
        SU[i] = SV[i] = LV[i] = 0;
        for (int j = 0; j < SIZE; j++)
        {
            slack[i * SIZE + j] = C[i * SIZE + j];
            X[i * SIZE + j] = 0;
        }
    }
    // cout<<endl;
    // for (int i=0; i<SIZE; i++)
    //     cout<<u[i]<<" ";
    // cout<<endl;

    // for (int i=0; i<SIZE; i++)
    //     cout<<v[i]<<" ";
    // cout<<endl;
    // cout<<"\nAdding noise\n";

    for (int i=0; i<SIZE; i++)
    {
        v[i] += deltaC[i];
        if (i>0)
            u[i] -= deltaC[i];
    }

    /*
    
    [[0,1,2],
    [3,4,5],
    [6,7,8]]
    
    u= [0,1,3]
    v= [0,3,5]

    [[10,11,12],
    [3,4,5],
    [6,7,8]]

    u = [10,13,15]
    v = [0,-9,-7]
    
    */

    // for (int i=1; i<SIZE; i++)
    // {
    //     u[i] -= deltaC[i];
    // }

    // for (int i=0; i<SIZE; i++)
    //     cout<<u[i]<<" ";
    // cout<<endl;

    // for (int i=0; i<SIZE; i++)
    //     cout<<v[i]<<" ";
    // cout<<endl;

    for (int i = 0; i < SIZE; i++)
    {
        X[rows[i] * SIZE + i] = 1;
        for (int j = 0; j < SIZE; j++)
            slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
    }
    int delta = -1;

    // Remove commenting to print the arrays

    /*
        cout<<"u:\n";
        print1DArray(u, SIZE);
        cout<<"v:\n";
        print1DArray(v, SIZE);
        cout<<"rows:\n";
        print1DArray(rows, SIZE);
        cout<<"phi:\n";
        print1DArray(phi, SIZE);
        cout<<"pred:\n";
        print1DArray(pred, SIZE);
        cout<<"SU:\n";
        print1DBoolArray(SU, SIZE);
        cout<<"SV:\n";
        print1DBoolArray(SV, SIZE);
        cout<<"LV:\n";
        print1DBoolArray(LV, SIZE);
        cout<<"X:\n";
        print2DArray(X, SIZE);
        cout<<"slack:\n";
        print2DArray(slack, SIZE);

    */

    int minval = infi;
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

    // Remove commenting to print the values of k and l
    /*   cout<<"k : "<<k<<endl<<"l : "<<l<<endl; */

    while (arrminval(slack, SIZE) < 0 && terminate == false)
    {
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
                    if (slack[i * SIZE + j] < minval)
                    {
                        minval = slack[i * SIZE + j];
                        k = i;
                        l = j;
                    }

            if (slack[k * SIZE + l] >= 0)
                terminate = true;
        }
        else
        {
            delta = dmin(SU, LV, C, u, v, SIZE);
            if (delta == infi)
                delta = u[k] + v[l] - C[k * SIZE + l];

            for (int i = 0; i < SIZE; i++)
                if (SU[i] == 1)
                    u[i] += delta;

            for (int j = 0; j < SIZE; j++)
                if (LV[j] == 1)
                    v[j] -= delta;

            if (u[k] + v[l] <= C[k * SIZE + l])
                for (int i = 0; i < SIZE; i++)
                    for (int j = 0; j < SIZE; j++)
                        slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];

            if (slack[k * SIZE + l] >= 0)
            {
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
            }
        }
    }

    // Remove comments to console print
    /*
        cout<<"Cost matrix C:\n";
        print2DArray(C, SIZE);

        cout<<"Assignment matrix X:\n";
        print2DArray(X, SIZE);

    */

    // outputFile << "Cost matrix C:\n";
    // printfile2DArray(C, outputFile, SIZE);

    // outputFile << "Assignment matrix X:\n";
    // printfile2DArray(X, outputFile, SIZE);

    int obj = 0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            obj += C[i * SIZE + j] * X[i * SIZE + j];

    cout << "Obj : " << obj << endl;
    // outputFile << "Obj : " << obj << endl;

    for (int i=0; i<SIZE; i++)
    {
        uvrowpred[i] = u[i];
        uvrowpred[i+SIZE]=v[i];
        uvrowpred[i+2*SIZE]=rows[i];
        uvrowpred[i+3*SIZE]=pred[i];
    }
    

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
    // delete[] uvrowpred;

    return uvrowpred;
}