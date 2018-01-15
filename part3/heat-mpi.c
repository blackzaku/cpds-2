/*
 * Iterative solver for heat distribution
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

void usage(char *s) {
    fprintf(stderr,
            "Usage: %s <input file> [result file]\n\n", s);
}

void save_to_file(char *filename, double *u, int np_rows, int np_columns, int nprocs) {
    double *uvis=(double*)calloc( sizeof(double), ((256/nprocs)+2) * (256+2) );
    FILE *auxfile = fopen(filename, "w");
    coarsen(u, np_columns, np_rows, uvis, 256 + 2, (256/nprocs) + 2);
    write_image(auxfile, uvis, 256 + 2, (256/nprocs) + 2);
}

int run_loop(int algorithm, double *u, double *uhelp, int np_rows, int np_columns, double *all_residual, int maxiter,
                     int myid, int numprocs) {
    MPI_Request request;
    double residual;
    int iter = 0;
    while (1) {
        switch (algorithm) {
            case 0: // JACOBI
                residual = relax_jacobi(u, uhelp, np_rows, np_columns);
                // Copy uhelp into u
                for (int i = 0; i < np_rows; i++)
                    for (int j = 0; j < np_columns; j++)
                        u[i * np_columns + j] = uhelp[i * np_columns + j];
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(u, np_rows, np_columns); // np_rows, np_columns);
                break;
            case 2: // GAUSS
                residual = relax_gauss(u, np_rows, np_columns);
                break;
        }

        iter++;

        MPI_Allreduce(&residual, all_residual, numprocs, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // solution good enough ?
        if (*all_residual < 0.00005) break;

        // max. iteration reached ? (no limit with maxiter=0)
        if (maxiter > 0 && iter >= maxiter) break;

        // SEND
        if (myid > 0) {
            MPI_Isend(&u[np_columns], np_columns, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, &request);
            // MPI_Isend(&uhelp[0], np_columns, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, 0);
        }
        if (myid < numprocs - 1) {
            MPI_Isend(&u[np_columns * (np_rows - 2)], np_columns, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD, &request);
            // MPI_Isend(&uhelp[np_columns*(np_rows-1)], np_columns, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD, 0);
        }
        // RECEIVE
        if (myid > 0) {
            MPI_Recv(&u[0], np_columns, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, 0);
            // MPI_Isend(&uhelp[0], np_columns, MPI_DOUBLE, myid - 1, 0, MPI_COMM_WORLD, 0);
        }
        if (myid < numprocs - 1) {
            MPI_Recv(&u[np_columns * (np_rows - 1)], np_columns, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD, 0);
            // MPI_Isend(&uhelp[np_columns*(np_rows-1)], np_columns, MPI_DOUBLE, myid + 1, 0, MPI_COMM_WORLD, 0);
        }
    }

    return iter;
}

int main(int argc, char *argv[]) {
    unsigned iter;
    FILE *infile, *resfile;
    char *resfilename;
    int myid, numprocs;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    if (myid == 0) {
        printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", myid, numprocs - 1);

        // algorithmic parameters
        algoparam_t param;
        int np, np_rows, np_columns, rows;

        double runtime, flop;
        double residual = 0.0;

        // check arguments
        if (argc < 2) {
            usage(argv[0]);
            return 1;
        }

        // check input file
        if (!(infile = fopen(argv[1], "r"))) {
            fprintf(stderr,
                    "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);

            usage(argv[0]);
            return 1;
        }

        // check result file
        resfilename = (argc >= 3) ? argv[2] : "heat.ppm";

        if (!(resfile = fopen(resfilename, "w"))) {
            fprintf(stderr,
                    "\nError: Cannot open \"%s\" for writing.\n\n",
                    resfilename);
            usage(argv[0]);
            return 1;
        }

        // check input
        if (!read_input(infile, &param)) {
            fprintf(stderr, "\nError: Error parsing input file.\n\n");
            usage(argv[0]);
            return 1;
        }
        print_params(&param);

        // set the visualization resolution

        param.u = 0;
        param.uhelp = 0;
        param.uvis = 0;
        param.visres = param.resolution;

        if (!initialize(&param)) {
            fprintf(stderr, "Error in Solver initialization.\n\n");
            usage(argv[0]);
            return 1;
        }

        // full size (param.resolution are only the inner points)
        np = param.resolution + 2;
        rows = param.resolution/numprocs;
        np_columns = param.resolution + 2;
        np_rows = rows + 2;

        // save_to_file("original.ppm", param.u, param.visres + 2, param.visres + 2, 1); // DEBUG

        // starting time
        runtime = wtime();

        // send to workers the necessary data to perform computation
        for (int i = 0; i < numprocs; i++) {
            if (i > 0) {
                MPI_Send(&param.maxiter, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.resolution, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.algorithm, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.u[np_columns * rows * i], np_rows * np_columns, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                MPI_Send(&param.uhelp[np_columns * rows * i], np_rows * np_columns, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }
        }


        iter=run_loop(param.algorithm, param.u, param.uhelp, np_rows, np_columns , &residual, param.maxiter, myid, numprocs);

        // receive results from workers
        for (int i = 0; i < numprocs; i++) {
            if (i > 0) {
                MPI_Recv(&param.u[np_columns * (rows * i + 1)], rows * np_columns, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            }
        }


        // Flop count after iter iterations
        flop = iter * 11.0 * param.resolution * param.resolution;
        // stopping time
        runtime = wtime() - runtime;

        fprintf(stdout, "Time: %04.3f ", runtime);
        fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n",
                flop / 1000000000.0,
                flop / runtime / 1000000);
        fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

        // for plot...
        coarsen(param.u, np, np,
                param.uvis, param.visres + 2, param.visres + 2);

        write_image(resfile, param.uvis,
                    param.visres + 2,
                    param.visres + 2);

        finalize(&param);

        fprintf(stdout, "Process %d finished computing with residual value = %f\n", myid, residual);

        MPI_Finalize();

        return 0;

    } else {

        printf("I am worker %d and ready to receive work to do ...\n", myid);

        // receive information from master to perform computation locally

        int columns, rows, np_rows, np_columns;
        int iter, maxiter;
        int algorithm;
        double residual = 0;


        MPI_Recv(&maxiter, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&columns, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&algorithm, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        rows = columns / numprocs;

        np_columns = columns + 2;
        np_rows = rows + 2;

        // allocate memory for worker
        double *u = calloc(sizeof(double), np_rows * np_columns);
        double *uhelp = calloc(sizeof(double), np_rows * np_columns);
        if ((!u) || (!uhelp)) {
            fprintf(stderr, "Error: Cannot allocate memory\n");
            return 0;
        }

        // fill initial values for matrix with values received from master
        MPI_Recv(&u[0], np_rows*np_columns, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&uhelp[0], np_rows*np_columns, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

        // save_to_file("child.ppm", u, np_rows, np_columns, numprocs); // DEBUG

        iter=run_loop(algorithm, u, uhelp, np_rows, np_columns, &residual, maxiter, myid, numprocs);

        // save_to_file("child_after.ppm", u, np_rows, np_columns, numprocs); // DEBUG

        MPI_Send(&u[np_columns], rows * np_columns, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        if (u) free(u);
        if (uhelp) free(uhelp);

        fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", myid, iter, residual);

        MPI_Finalize();
        exit(0);
    }
}

