#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define EPSILON 0.00001

double* initB(int matrixSize){
    double* b = (double*) calloc(matrixSize, sizeof(double));
    b[0] = 20;
    b[1] = 40;
    b[2] = -30;
    return b; 
}

double* initX(int matrixSize){
    double* x = (double*) calloc(matrixSize, sizeof(double));
    return x;
}

double* initMatrix(int matrixSize){
    int bigMatrixSize = matrixSize * matrixSize;
    double* matrix = (double*)calloc(bigMatrixSize * bigMatrixSize, sizeof(double));
    for(int i = 0; i < matrixSize; ++i){
        int shiftFromZero = i * matrixSize  + i * matrixSize * bigMatrixSize;
        for(int j = 0; j < matrixSize; ++j){
            int index = shiftFromZero + bigMatrixSize * j + j;
            matrix[index] = -4;
        }
        for(int j = 1; j < matrixSize; ++j){
            int topStringIndex = bigMatrixSize * (j - 1) + j;
            int bottomStringIndex = bigMatrixSize * j  + (j - 1);
            matrix[shiftFromZero + topStringIndex ] = 1;
            matrix[shiftFromZero  + bottomStringIndex ] = 1;
        }
    }
    for(int i = matrixSize; i < bigMatrixSize; ++i){
            int topStringIndex = bigMatrixSize * (i - matrixSize) + i;
            int bottomStringIndex = bigMatrixSize * i  + (i - matrixSize);
            matrix[topStringIndex] = 1;
            matrix[bottomStringIndex] = 1;
    }
    return matrix; 
}

double* powMatrixCol(double* matrix, double* column, int size){
    double* C = (double*)calloc(size, sizeof(double));

    #pragma omp parallel for  
    for (int i = 0; i < size; ++i){
        for (int j = 0 ; j < size; ++j){
            C[i] += matrix[i * size + j] *column[j];
        }

    }
	return C;

}

double* powConfCol(double conf, double* column, int size){
    double* C = (double*)calloc(size, sizeof(double));

    #pragma omp parallel for  
    for (int i = 0; i < size; ++i){
        C[i] = column[i] * conf;
    }
	return C;
}


double* minusColumn(double* A, double*B, int size){
    double* C = (double*)malloc(size  * sizeof(double));

    #pragma omp parallel for  
    for(int i = 0; i < size; ++i){
        C[i] = A[i] - B[i];
    }
    return C;
}

double* plusColumn(double* A, double* B, int size){
    double* C = (double*)malloc(size  * sizeof(double));

    #pragma omp parallel for  
    for(int i = 0; i < size; ++i){
        C[i] = A[i] + B[i];
    }
    return C;
}



double scalarPow(double* A, double* B, int size){
    double result = 0;

    #pragma omp parallel for reduction(+:result)   
    for(int i = 0; i < size; ++i){
        result += A[i] * B[i];
    }
    return result;
}

double norma(double* vector, int size){
    double norma  = 0;

    #pragma omp parallel for reduction(+:norma)  
    for(int i = 0; i < size; ++i){
        norma += vector[i] * vector[i];
    }
    return sqrt(norma);
}

char notGoodApproaching(double* r0, double b0Norma, int sizeMatrix){
    double result =  norma(r0, sizeMatrix) / b0Norma;
    if(result < EPSILON){
        return 0;
    }
    return 1;
    
}

double getAlpha(double* r, double* matrix, double* z, int sizeMatrix){
    double upperScalar = scalarPow(r, r, sizeMatrix);
    double* pow = powMatrixCol(matrix, z, sizeMatrix);
    double bottomScalar = scalarPow(pow, z, sizeMatrix);
    free(pow);
    double result = upperScalar/bottomScalar;
    return result;
}

double getBetta(double* r1, double* r0, int sizeMatrix){
    double upperScalar = scalarPow(r1, r1, sizeMatrix);
    double bottomScalar = scalarPow(r0, r0, sizeMatrix);
    double result = upperScalar/bottomScalar;
    return result;
}

double* systemSolution(double* matrix, double* b, double* x, int sizeMatrix){
    
    double* pow = powMatrixCol(matrix, x, sizeMatrix);
    double* r0 = minusColumn(b, pow, sizeMatrix);
   
    double* z0 = (double*)malloc(sizeMatrix * sizeof(double));
    #pragma omp parallel for  
    for(int i = 0; i < sizeMatrix; ++i){
        z0[i] = r0[i];
    }
    double alpha1 = getAlpha(r0, matrix, z0, sizeMatrix);
    double betta1 = 0;
    double b0Norma = norma(b, sizeMatrix);
    while(notGoodApproaching(r0, b0Norma, sizeMatrix) == 1){
        double* pcc= powConfCol(alpha1, z0, sizeMatrix);
        double* x1 = plusColumn(x, pcc, sizeMatrix);
        free(pcc);
        double* pwc = powMatrixCol(matrix, z0, sizeMatrix);
        pcc = powConfCol(alpha1, pwc, sizeMatrix);
        free(pwc);
        double* r1 = minusColumn(r0, pcc, sizeMatrix);
        free(pcc);
        betta1 = getBetta(r1, r0, sizeMatrix);
        pcc = powConfCol(betta1, z0, sizeMatrix);
        double* z1 = plusColumn(r1, pcc, sizeMatrix);
        free(pcc);
        free(x);
        free(r0);
        free(z0);
        x = x1;
        r0 = r1;
        z0 = z1;
        alpha1 = getAlpha(r0, matrix, z0, sizeMatrix);
    }
    free(r0);
    free(pow);
    free(z0);
    return x;
}

void printSol(double* sol, int size){
    for(int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            printf("%f ", sol[i * size + j]);   
        }
        printf("\n");

    }
}

int main(){
    omp_set_num_threads(4);
    double startTime, endTime;
    startTime = omp_get_wtime();
    int sizeMatrix = 100;
    int sizeBigMatrix = sizeMatrix * sizeMatrix;
    double* bigMatrix = initMatrix(sizeMatrix);
    double* b = initB(sizeBigMatrix );
    double* sol = initX(sizeBigMatrix );
    sol = systemSolution(bigMatrix, b, sol, sizeBigMatrix );
    //printSol(sol, sizeMatrix);
    free(b);
    free(bigMatrix);
    free(sol);
    endTime = omp_get_wtime();
	double workingTime = (endTime - startTime);
	printf("working time %f \n", workingTime);
}
