#include "heat.h"

#define NB 8

#define min(a,b) ( ((a) < (b)) ? (a) : (b) )

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi (double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum=0.0;
    int nbx, bx, nby, by;
  
    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    
    // The outermost loops (ii, jj) iterates over blocks of nbx*nby
    // We can optimize the value of NB
    #pragma omp parallel for collapse(2) private(diff) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            utmp[i*sizey+j]= 0.25 * (u[ i*sizey     + (j-1) ]+  // left
					     u[ i*sizey     + (j+1) ]+  // right
				             u[ (i-1)*sizey + j     ]+  // top
				             u[ (i+1)*sizey + j     ]); // bottom
	            diff = utmp[i*sizey+j] - u[i*sizey + j];
	            sum += diff * diff; 
	        }

    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 * BRBRBRBR...
 * RBRBRBRB...
 * BRBRBRBR...
 * ...
 */
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    // Computing "Red" blocks
    #pragma omp parallel
    {
        #pragma omp for private(unew) reduction(+:sum)
        for (int ii=0; ii<nbx; ii++) {
            lsw = ii%2;
            for (int jj=lsw; jj<nby; jj=jj+2) 
                for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                    for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	                unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				          u[ i*sizey	+ (j+1) ]+  // right
				          u[ (i-1)*sizey	+ j     ]+  // top
				          u[ (i+1)*sizey	+ j     ]); // bottom
	                diff = unew - u[i*sizey+ j];
	                sum += diff * diff; 
	                u[i*sizey+j]=unew;
	            }
        }

        // Computing "Black" blocks
        #pragma omp for private(unew) reduction(+:sum)
        for (int ii=0; ii<nbx; ii++) {
            lsw = (ii+1)%2;
            for (int jj=lsw; jj<nby; jj=jj+2) 
                for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                    for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	                unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				          u[ i*sizey	+ (j+1) ]+  // right
				          u[ (i-1)*sizey	+ j     ]+  // top
				          u[ (i+1)*sizey	+ j     ]); // bottom
	                diff = unew - u[i*sizey+ j];
	                sum += diff * diff; 
	                u[i*sizey+j]=unew;
	            }
        }
    }
    return sum;
}
/*
double relax_redblack (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    // Computing "Red" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = ii%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    // Computing "Black" blocks
    for (int ii=0; ii<nbx; ii++) {
        lsw = (ii+1)%2;
        for (int jj=lsw; jj<nby; jj=jj+2) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
	        }
    }

    return sum;
}
*/


/*
 * Blocked Gauss-Seidel solver: one iteration step
 * If not using the task model use block[ii][jj]=1 and while (!block[ii][jj]); - flush directive
 * Check whats the meaning of the volatile keyword in C++
 */
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;
    
    int semaphores[NB][NB] = {};
    
    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    #pragma omp parallel for private(diff) reduction(+:sum)
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++)
        {
            #pragma omp flush
            while ((ii > 0 && semaphores[ii - 1][jj] == 0) ||
                   (jj > 0 && semaphores[ii][jj - 1] == 0))
            {
                #pragma omp flush
            }
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
                }
            semaphores[ii][jj] = 1;
            #pragma omp flush
            }

    return sum;
}
/*
double relax_gauss (double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum=0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex/nbx;
    nby = NB;
    by = sizey/nby;
    for (int ii=0; ii<nbx; ii++)
        for (int jj=0; jj<nby; jj++) 
            for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) 
                for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
	            unew= 0.25 * (    u[ i*sizey	+ (j-1) ]+  // left
				      u[ i*sizey	+ (j+1) ]+  // right
				      u[ (i-1)*sizey	+ j     ]+  // top
				      u[ (i+1)*sizey	+ j     ]); // bottom
	            diff = unew - u[i*sizey+ j];
	            sum += diff * diff; 
	            u[i*sizey+j]=unew;
                }

    return sum;
}
*/
