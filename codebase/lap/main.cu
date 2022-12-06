#include "include/main.h"

int main(int argc, char **argv)
{
    const int seed = 45345;
    int N = atoi(argv[1]);
    double range = strtod(argv[2], nullptr);
    const int n_tests = 1;
    int N2 = N * N;
    int devid = 1;
    int spcount = 1;
    printf("range: %f\n", range);
    double *C = new double[N2];
    // C = read_normalcosts(C, &N, filepath);
    range *= N;
    printf("range: %f\n", range);

    default_random_engine generator(seed);
    uniform_int_distribution<int> distribution(0, range - 1);
    // uniform_real_distribution<double> distribution(0, range - 1);
    long long total_time = 0;
    for (int test = 0; test < n_tests; test++)
    {
        double *C = generate_cost<double>(N, range / N);

        cudaSafeCall(cudaGetDevice(&devid), "cuda device unavailable!", __LINE__, __FILE__);
        // printHostMatrix(C, N, N, "LAP costs as read");
        typedef std::chrono::high_resolution_clock clock;
        auto start = clock::now();
        double *d_C = nullptr;
        int *d_row_assignments = nullptr;
        double *d_row_duals = nullptr;
        double *d_col_duals = nullptr;
        double *d_obj = nullptr;

        cudaSafeCall(cudaMalloc((void **)&d_C, spcount * N * N * sizeof(double)), "Error in cudaMalloc d_costs", __LINE__, __FILE__);
        cudaSafeCall(cudaMalloc((void **)&d_row_assignments, spcount * N * sizeof(int)), "Error in cudaMalloc d_costs", __LINE__, __FILE__);
        cudaSafeCall(cudaMalloc((void **)&d_row_duals, spcount * N * sizeof(double)), "Error in cudaMalloc d_costs", __LINE__, __FILE__);
        cudaSafeCall(cudaMalloc((void **)&d_col_duals, spcount * N * sizeof(double)), "Error in cudaMalloc d_costs", __LINE__, __FILE__);
        cudaSafeCall(cudaMalloc((void **)&d_obj, spcount * sizeof(double)), "Error in cudaMalloc d_costs", __LINE__, __FILE__);
        cudaSafeCall(cudaMemcpy(d_C, C, N * N * sizeof(double), cudaMemcpyDefault), "Error at ", __LINE__, __FILE__);

        CuLAP LAP(N, spcount, devid, false);

        LAP.solve(d_C, d_row_assignments, d_row_duals, d_col_duals, d_obj);
        auto elapsed = clock::now() - start;

        printDebugArray(d_obj, 1, "Total cost:");
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        total_time += microseconds / n_tests;

        // printDebugArray(d_row_assignments, N, "LAP assignments:");
        cudaFree(d_C);
        cudaFree(d_row_assignments);
        cudaFree(d_row_duals);
        cudaFree(d_col_duals);
        cudaFree(d_obj);
    }

    cout << "Time taken: \t" << total_time / 1000.0f << " ms" << endl;
    // printDebugArray(d_row_assignments, N, "LAP assignments:");
    return 0;
}
