#include <iostream>
#include <cstdlib>
#include <ctime>
#include <fstream>

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

        float values[SIZE][SIZE] = {
            {7, 9, 8, 9},
            {2, 8, 5, 7},
            {1, 6, 6, 9},
            {3, 6, 2, 2}};

        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                C[i * SIZE + j] = values[i][j];

        for (int i = 0; i < SIZE; ++i)
        {
            for (int j = 0; j < SIZE; ++j)
                cout << C[i * SIZE + j] << " ";
            cout << endl;
        }
        return C;
    }
    else if (arrChoice == 2)
    {
        SIZE = 4;
        float *C = new float[SIZE * SIZE];
        float values[SIZE][SIZE] = {
            {3, 8, 2, 1},
            {2, 7, 5, 5},
            {9, 8, 1, 2},
            {1, 8, 5, 3}};

        for (int i = 0; i < SIZE; ++i)
            for (int j = 0; j < SIZE; ++j)
                C[i * SIZE + j] = values[i][j];

        for (int i = 0; i < SIZE; ++i)
        {
            for (int j = 0; j < SIZE; ++j)
                cout << C[i * SIZE + j] << " ";
            cout << endl;
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
    int *phi = new int[SIZE]; // Used for backtracking. Index:Row and Value:Column
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
                if (near_zero(C[node * SIZE + j] - u[node] - v[j]) == 1)
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
                if (near_zero(C[i * SIZE + j] - u[i] - v[j]) == 1)
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

    delete[] visited;
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

void initializeMemory(int *&pred,
                      bool *&SU, bool *&SV, bool *&LV,
                      float *&slack, int SIZE)
{

    pred = new int[SIZE]; // To keep track of new assignments
    SU = new bool[SIZE];  // Nodes scanned on LHS of the bipartite graph
    SV = new bool[SIZE];  // Nodes scanned on RHS of the bipartite graph
    LV = new bool[SIZE];  // Bookkeeping to scan the nodes one by one on RHS

    slack = new float[SIZE * SIZE]; // Reduced cost matrix
}

void cleanupMemory(int *&pred,
                   bool *&SU, bool *&SV, bool *&LV,
                   float *&slack)
{

    delete[] pred;
    delete[] SU;
    delete[] SV;
    delete[] LV;

    delete[] slack;
}

int bal_common(
    float *C, float *u, float *v, int SIZE, float *slack,
    bool *SU, bool *SV, bool *LV,
    int *rows, int *pred)
{
    for (int i = 0; i < SIZE; i++)
        SU[i] = SV[i] = LV[i] = 0;

    float delta = -1;
    float minval = infi;
    int k = -1;
    int l = -1;

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (slack[i * SIZE + j] <= minval)
            {
                minval = slack[i * SIZE + j];
                k = i;
                l = j;
            }

    bool terminate = false;

    int counter = 0;

    while (arrminval(slack, SIZE) < -eps && terminate == false)
    {
        // cout << "\033[1;32m";
        // cout << "Minimum value : " << arrminval(slack, SIZE) << endl;
        // cout << "\033[0m";

        counter++;
        if (counter % 1000 == 0)
            cout << counter << endl;
        else if (counter > 1e6)
        {
            exit(-1);
        }
        bool path_found = alternating(k, l, SU, SV, LV, rows, pred, C, u, v, SIZE);

        if (path_found == true)
        {
            for (int i = 0; i < SIZE; i++)
                rows[i] = pred[i];

            rows[l] = k;

            delta = u[k] + v[l] - C[k * SIZE + l];
            v[l] -= delta;

            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];

            minval = infi;
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    if (slack[i * SIZE + j] <= minval)
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
    // cout << "\033[1;31m";
    // cout << "Exit Minimum value : " << arrminval(slack, SIZE) << endl;
    // cout << "\033[0m";

    float obj = 0.0;
    for (int i = 0; i < SIZE; i++)
        obj += C[rows[i] * SIZE + i];

    cout << "\033[1;32m";
    cout << "Balinski Counter : " << counter << endl;
    cout << "Balinski Objective : " << obj << endl;
    cout << "\033[0m";

    // cout << "\033[1;33m";
    // cout << "rows : ";
    // for (int i = 0; i < SIZE; i++)
    //     cout << rows[i] << " ";
    // cout << endl;
    // cout << "\033[0m";

    // cout << "\033[1;33m";
    // cout << "u after : ";
    // for (int i = 0; i < SIZE; i++)
    //     cout << u[i] << " ";
    // cout << endl;

    // cout << "v after : ";
    // for (int i = 0; i < SIZE; i++)
    //     cout << v[SIZE + i] << " ";
    // cout << endl;
    // cout << "\033[0m";

    return 0;
}

int balinski_solve(float *C, int SIZE)
{
    float *u, *v;
    int *rows, *pred;
    bool *SU, *SV, *LV;
    float *slack;
    initializeMemory(pred,
                     SU, SV, LV, slack, SIZE);

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
        }
    }

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
            slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
    }

    bal_common(C, u, v, SIZE, slack,
               SU, SV, LV,
               rows, pred);

    cleanupMemory(pred, SU, SV, LV, slack);

    return 0;
}

int balinski_resolve(float *C, int SIZE, float *u, float *v, int *rows, float *NC, int precision, int disp_C)
{
    // float *u, *v;
    int *pred;
    bool *SU, *SV, *LV;
    float *slack;

    initializeMemory(pred,
                     SU, SV, LV, slack, SIZE);

    /*
    Adding noise to the new Cost Matrix and preserving original
    */
    float *origC = new float[SIZE * SIZE];
    float *deltaC = new float[SIZE * SIZE];

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            origC[i * SIZE + j] = C[i * SIZE + j];

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            C[i * SIZE + j] += floor(NC[i * SIZE + j] * C[i * SIZE + j] * pow(10, precision)) / pow(10, precision);

    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            deltaC[i * SIZE + j] = C[i * SIZE + j] - origC[i * SIZE + j];

    if (disp_C == 2)
    {
        cout << "\033[1;33m";
        cout << "Orig C Matrix : " << endl;
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
                cout << origC[i * SIZE + j] << " ";
            cout << endl;
        }
        cout << "\033[0m";

        cout << "New C Matrix : " << endl;
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
                cout << C[i * SIZE + j] << " ";
            cout << endl;
        }
        cout << "\033[1;33m";
        cout << "Delta C Matrix : " << endl;
        for (int i = 0; i < SIZE; i++)
        {
            for (int j = 0; j < SIZE; j++)
                cout << deltaC[i * SIZE + j] << " ";
            cout << endl;
        }
    }

    /*
    Retrieve old duals and assignments
    */
    for (int i = 0; i < SIZE; i++)
    {
        pred[i] = rows[i];
        for (int j = 0; j < SIZE; j++)
        {
            slack[i * SIZE + j] = C[i * SIZE + j];
        }
    }

    // cout << "\033[1;32m";
    // cout << "Starting rows : ";
    // for (int i = 0; i < SIZE; i++)
    //     cout << rows[i] << " ";
    // cout << endl;

    // cout << "\033[1;33m";
    // cout << "Starting u : ";
    // for (int i = 0; i < SIZE; i++)
    //     cout << u[i] << " ";
    // cout << endl;

    // cout << "Starting v : ";
    // for (int i = 0; i < SIZE; i++)
    //     cout << v[i] << " ";
    // cout << endl;
    // cout << "\033[0m";

    for (int i = 0; i < SIZE; i++)
    {
        u[rows[i]] += ceil(deltaC[rows[i] * SIZE + i] / 2 * pow(10, precision)) / pow(10, precision);
        v[i] += floor(deltaC[rows[i] * SIZE + i] / 2 * pow(10, precision)) / pow(10, precision);
    }

    // cout << "u new : ";
    // for (int i = 0; i < SIZE; i++)
    //     cout << u[i] << " ";
    // cout << endl;

    // cout << "v new : ";
    // for (int i = 0; i < SIZE; i++)
    //     cout << v[i] << " ";
    // cout << endl;

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
            slack[i * SIZE + j] = C[i * SIZE + j] - u[i] - v[j];
    }

    bal_common(C, u, v, SIZE, slack,
               SU, SV, LV,
               rows, pred);

    cleanupMemory(pred, SU, SV, LV, slack);

    return 0;
}