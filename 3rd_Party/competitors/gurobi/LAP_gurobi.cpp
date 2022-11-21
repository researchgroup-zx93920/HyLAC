#include <gurobi_c++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <random>
#include <stdlib.h>

using namespace std;
using namespace chrono;

double *read_normalcosts(double *C, int *Nad, const char *filepath)
{
    string s = filepath;
    ifstream myfile(s.c_str());
    if (!myfile)
    {
        std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
        exit(-1);
    }
    myfile >> Nad[0];
    int N = Nad[0];
    C = new double[N * N];
    for (int i = 0; i < N * N; i++)
    {
        myfile >> C[i];
    }
    myfile.close();
    return C;
}

int main(int argc, char **argv)
{

    const int seed = 45345;
    int N = atoi(argv[1]);

    double range = strtod(argv[2], nullptr);
    int N2 = N * N;
    // const char *filepath = argv[1];
    // C = read_normalcosts(C, &N, filepath);
    double *C = new double[N2];
    range *= N;

    default_random_engine generator(seed);
    uniform_int_distribution<int> distribution(0, range - 1);

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < N; k++)
        {
            double gen = (double)distribution(generator);
            // cout << gen << "\t";
            C[N * i + k] = gen;
        }
        // cout << endl;
    }
    // for (int i = 0; i < N; i++)
    // {
    //     for (int k = 0; k < N; k++)
    //     {
    //         cout << C[N * i + k] << "\t";
    //     }
    //     cout << endl;
    // }

    try
    {
        GRBEnv env = GRBEnv();
        GRBModel model = GRBModel(env);
        GRBVar *x = new GRBVar[N2];
        for (int i = 0; i < N; i++)
        {
            for (int k = 0; k < N; k++)
            {

                // costx represents the coefficient of x variables in the objective function
                double costx = C[N * i + k];
                stringstream s;
                s << "X_" << i << "_" << k << endl;
                x[N * i + k] = model.addVar(0.0, 1.0, costx, GRB_CONTINUOUS, s.str());
            }
        }
        model.update();
        for (int k = 0; k < N; k++)
        {
            GRBLinExpr lhs = 0;
            for (int i = 0; i < N; i++)
            {
                lhs += x[N * i + k];
            }
            stringstream s;
            s << "XR_" << k << endl;
            model.addConstr(lhs == 1, s.str());
        }
        model.update();

        for (int i = 0; i < N; i++)
        {
            GRBLinExpr lhs = 0;
            for (int k = 0; k < N; k++)
            {
                lhs += x[N * i + k];
            }
            stringstream s;
            s << "XR1_" << i << endl;
            model.addConstr(lhs == 1, s.str());
        }
        model.update();
        auto start = high_resolution_clock::now();
        model.optimize();
        auto elapsed = high_resolution_clock::now() - start;

        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();

        cout << "Objective: " << model.getObjective().getValue() << endl;
        cout << "Time: " << microseconds / 1000.0f << " ms" << endl;
    }
    catch (GRBException e)
    {
        cout << "Error code = "
             << e.getErrorCode()
             << endl;
        cout << e.getMessage() << endl;
    }
    catch (...)
    {
        cout << "Exception during optimization"
             << endl;
    }

    return 0;
}