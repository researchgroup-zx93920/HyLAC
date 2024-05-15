#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

const float infi = 100000.0;
float eps = 1e-4;

int SIZE = 10;

bool near_zero(float val)
{
    return ((val < eps) && (val > -eps));
}

float arrminval(float *arr, int SIZE)
{
    float minimum = infi;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            if (arr[i * SIZE + j] < minimum)
                minimum = arr[i * SIZE + j];
    printf("Minimum value : %f\n", minimum);
    return minimum;
}

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

void initializeMemory(float *&u, float *&v,
                      int *&rows, int *&pred,
                      bool *&SU, bool *&SV, bool *&LV, bool *&X,
                      float *&slack, int SIZE)
{
    u = new float[SIZE];            // Dual for row
    v = new float[SIZE];            // Dual for column
    rows = new int[SIZE];           // To keep track of assigned nodes. Indices: RHS, Values: LHS of Bipartite graph
    pred = new int[SIZE];           // To keep track of new assignments
    SU = new bool[SIZE];            // Nodes scanned on LHS of the bipartite graph
    SV = new bool[SIZE];            // Nodes scanned on RHS of the bipartite graph
    LV = new bool[SIZE];            // Bookkeeping to scan the nodes one by one on RHS
    X = new bool[SIZE * SIZE];      // Assignment matrix
    slack = new float[SIZE * SIZE]; // Reduced cost matrix
}

void cleanupMemory(float *&u, float *&v,
                   int *&rows, int *&pred,
                   bool *&SU, bool *&SV, bool *&LV, bool *&X,
                   float *&slack)
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

int *bal_common(
    float *C, float *u, float *v, int SIZE, float *slack,
    bool *SU, bool *SV, bool *LV,
    bool *X, int *rows, int *pred)
{
    int *uvrowc = new int[SIZE * 3];
    for (int i = 0; i < SIZE; i++)
        SU[i] = SV[i] = LV[i] = 0;

    float checksum = 0;
    for (int i = 0; i < SIZE * SIZE; i++)
        checksum += C[i];
    cout << "Check avg : " << checksum / (SIZE * SIZE) << endl;

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
        counter++;
        if (counter % 1000 == 0)
            cout << counter << endl;
        else if (counter > 1000)
        {
            exit(-1);
        }
        // print arrays u, v, row
        cout << "u : ";
        for (int i = 0; i < SIZE; i++)
            cout << u[i] << " ";
        cout << endl;

        cout << "v : ";
        for (int i = 0; i < SIZE; i++)
            cout << v[i] << " ";
        cout << endl;

        cout << "row : ";
        for (int i = 0; i < SIZE; i++)
            cout << rows[i] << " ";
        cout << endl;

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
                    if (slack[i * SIZE + j] <= minval)
                    {
                        minval = slack[i * SIZE + j];
                        k = i;
                        l = j;
                    }

            if (slack[k * SIZE + l] >= eps)
            {
                printf("Termination signalled at line: %u\n", __LINE__);
                terminate = true;
            }
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
                {
                    printf("Termination signalled at line: %u\n", __LINE__);
                    terminate = true;
                }
            }
        }
    }
    if (arrminval(slack, SIZE) >= 0)
        printf("Termination signalled at line: %u\n", __LINE__);
    float obj = 0.0;
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            obj += C[i * SIZE + j] * X[i * SIZE + j];

    cout << "Balinski Counter : " << counter << endl;
    cout << "Balinski Objective : " << obj << endl;

    for (int i = 0; i < SIZE; i++)
    {
        uvrowc[i] = u[i];
        uvrowc[i + SIZE] = v[i];
        uvrowc[i + 2 * SIZE] = rows[i];
    }

    cout << "\033[1;33m";
    // cout<<"Balinski slack : \n";
    // for (int i=0; i<SIZE; i++)
    // {
    //     for (int j=0; j<SIZE; j++)
    //         cout<<C[i*SIZE+j]-u[i]-v[j]<<" ";
    //     cout<<endl;
    // }
    cout << "\033[0m";
    return uvrowc;
}

int main()
{
    float *C = new float[SIZE * SIZE];
    float *u, *v;
    int *rows, *pred;
    bool *SU, *SV, *LV, *X;
    float *slack;

    initializeMemory(u, v, rows, pred, SU, SV, LV, X, slack, SIZE);
    int *uvrow = new int[SIZE * 3];
    // Ask the user for a seed value
    int seed = 65456;
    // cout << "Enter a seed value : ";
    // cin >> seed;
    srand(seed);

    // Generate a random matrix C of size SIZE x SIZE with values ranging from 1 to 100 and each value should have 1 decimal point
    // for (int i = 0; i < SIZE * SIZE; i++)
    //     C[i] = (float)(rand() % 100 + 1) + noise[i];

    ifstream file("cost10_float.csv");

    if (!file.is_open())
    {
        cerr << "Error opening file!" << endl;
        return 1;
    }

    string line;
    int row = 0;
    while (getline(file, line) && row < SIZE)
    {
        istringstream iss(line);
        string value;
        int col = 0;
        while (getline(iss, value, ','))
        {
            if (col < SIZE)
            {
                C[row * SIZE + col] = stof(value) * 1;
            }
            col++;
        }
        row++;
    }

    file.close();

    // Print matrix C
    for (size_t i = 0; i < SIZE; i++)
    {
        for (size_t j = 0; j < SIZE; j++)
        {
            cout << C[i * SIZE + j] << ", ";
        }
        cout << endl;
    }

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

    uvrow = bal_common(C, u, v, SIZE, slack, SU, SV, LV, X, rows, pred);

    cleanupMemory(u, v, rows, pred, SU, SV, LV, X, slack);

    delete[] C;
    delete[] uvrow;

    return 0;
}