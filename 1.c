
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#define EPSILON 0.00001

double *initB(int matrixSize)
{
    double *b = (double *)calloc(matrixSize, sizeof(double));
    b[0] = 20;
    b[1] = 40;
    b[2] = -30;
    return b;
}

double *initX(int matrixSize)
{
    double *x = (double *)calloc(matrixSize, sizeof(double));
    return x;
}

int returnstandartStrFromChunk(int size)
{
    int commRank, commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    return size / commSize;
}

int defineChunk(int size)
{
    int chunk;
    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    int strFromChunk = size / commSize;
    chunk = size * strFromChunk;
    if (commSize - 1 == commRank ){
        chunk += size * (size % commSize);
    }

    return chunk;
}


double *initMatrix(int matrixSize, int chunk, int standartStrFromChunk){ 
    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

    int bigMatrixSize = matrixSize * matrixSize;
    double* matrix = (double*)calloc(chunk, sizeof(double));
    int numberString = standartStrFromChunk * commRank;
    int countRight = standartStrFromChunk * commRank % matrixSize;
    int countLeft = (standartStrFromChunk * commRank - 1) %  matrixSize;
    if (countLeft  < 0){
        countLeft  = 0;
    }
    for(int i = 0; i < chunk / bigMatrixSize; ++i){
        matrix[i * bigMatrixSize + numberString] = -4;
        if(numberString + matrixSize  < bigMatrixSize){
            matrix[i * bigMatrixSize + numberString + matrixSize] = 1; 
        }
        if(numberString - matrixSize  >= 0){
            matrix[i * bigMatrixSize + numberString - matrixSize] = 1; 
        }
        if( countRight != matrixSize - 1){
            if(numberString + 1 < bigMatrixSize){
                countRight++;
                matrix[i * bigMatrixSize + numberString + 1] = 1;
            }
        }
        else{
            countRight = 0;
        }
        if( countLeft != matrixSize - 1){
            if(numberString - 1 >= 0){
                
                countLeft++;
                matrix[i * bigMatrixSize + numberString - 1] = 1;
            }
        }
        else{
            countLeft = 0;
        }
        numberString++;
    }
    return matrix; 
}

double* powMatrixCol(double *matrix, double *column, int size, int chunk, int shift)
{
    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    int strFromChunk = chunk / size;
    double *localC = (double *)calloc(strFromChunk, sizeof(double));
    for (int i = 0; i < strFromChunk; ++i)
    {
        for (int j = shift; j < size; ++j)
        {
            int matrixIndex = i * size + j;
            int columnIndex = j;
            localC[i] += matrix[matrixIndex] * column[columnIndex];
           
        }
    }
    return localC;
}

double *powConfCol(double conf, double *column, int size)
{
    double *localC = (double *)calloc(size, sizeof(double));
    for (int i = 0; i < size; ++i)
    {
        localC[i] = column[i] * conf;
        ;
    }
    return localC;
}

double *minusColumn(double *A, double *B, int size, int shift)
{
    int commRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    double *localC = (double *)calloc(size, sizeof(double));
    for (int i = shift; i < shift + size; ++i)
    {
        localC[i - shift] = A[i] - B[i - shift];
    }

    
    return localC;
}

double *plusColumn(double *A, double *B, int size, int shift)
{
    int commRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    double *localC = (double *)calloc(size, sizeof(double));
    for (int i = shift; i < shift + size; ++i)
    {
        localC[i - shift] = A[i] + B[i - shift];
    }
    return localC;
}

double scalarPow(double *A, double *B, int size)
{
    double result = 0;
    for (int i = 0; i < size; ++i)
    {
        result += A[i] * B[i];
    }
    return result;
}

double norma(double *vector, int size)
{
    double norma = 0;
    double mainNorma = 0;
    for (int i = 0; i < size; ++i)
    {
        norma += vector[i] * vector[i];
    }
    MPI_Reduce(&norma, &mainNorma, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mainNorma, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return sqrt(mainNorma);
}

char notGoodApproaching(double *r0, double b0Norma, int shortSize)
{   
    int commRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    double result = norma(r0, shortSize) / b0Norma;
    if (result < EPSILON)
    {
        return 0;
    }
    return 1;
}

double getAlpha(double *r, double *matrix, double *z, int size, int chunk)
{   
    int shortSize = chunk / size;
    double result;
    double mainUpperScalar, mainBottomScalar;
    int commSize, commRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    double upperScalar = scalarPow(r, r, shortSize);
    MPI_Reduce(&upperScalar, &mainUpperScalar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double* mainZ = (double*)malloc(size * sizeof(double));
    int* count = (int*)malloc(size * sizeof(int));
    for(int i = 0; i < size; ++i ){
        count[i] = chunk / size;
    }
    count[commSize - 1] += size % commSize;
    int* shift = (int*)malloc(commSize * sizeof(int));
    shift[0] = 0;
    for(int i = 1; i < commSize; ++i ){
        shift[i] = count[i-1]+shift[i-1];
    }
    MPI_Gatherv(z, chunk / size, MPI_DOUBLE, mainZ, count, shift, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(mainZ, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double* pow = powMatrixCol(matrix, mainZ, size, chunk,  0);
    double bottomScalar = scalarPow(pow, z, shortSize);
    MPI_Reduce(&bottomScalar, &mainBottomScalar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (commRank == 0)
    {
        result = mainUpperScalar / mainBottomScalar;
    }
    MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(shift);
    free(count);
    return result;
}

double getBetta(double *r1, double *r0, int shortSize)
{   
    double result  = 0;
    int commRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    double mainUpperScalar, mainBottomScalar;
    double upperScalar = scalarPow(r1, r1, shortSize);
    MPI_Reduce(&upperScalar, &mainUpperScalar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double bottomScalar = scalarPow(r0, r0, shortSize);
    MPI_Reduce(&bottomScalar, &mainBottomScalar, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (commRank == 0)
    {
        result = mainUpperScalar / mainBottomScalar;
    }
    MPI_Bcast(&result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return result;
}

double *systemSolution(double *matrix, double *b, double *x, int sizeMatrix, int chunk, int standartStrFromChunk){
    int commRank, commSize;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    int shortSize = chunk / sizeMatrix;
    int shift = commRank * shortSize;
    int sizeSkip = commRank * standartStrFromChunk;
    double* pow  = powMatrixCol(matrix, x, sizeMatrix, chunk, sizeSkip);
    double *r0 = minusColumn(b, pow, shortSize, shift);
    double *z0 = (double *)malloc(shortSize * sizeof(double));
    for (int i = 0; i < shortSize; ++i)    {
        z0[i] = r0[i];
    }
    double alpha1 = getAlpha(r0, matrix, z0, sizeMatrix, chunk);
    double betta1 = 0;
    double bNorma = 0;
    for (int i = 0; i < sizeMatrix; ++i){
        bNorma += b[i] * b[i];
    }

    bNorma = sqrt(bNorma);
    int count = 0;
    while (notGoodApproaching(r0, bNorma, shortSize) == 1 ){
        double* x1 = plusColumn(x, powConfCol(alpha1, z0, shortSize), shortSize, 0);
        double* mainZ = (double*)malloc(sizeMatrix * sizeof(double));

        int* count = (int*)malloc(sizeMatrix * sizeof(int));
        for(int i = 0; i < sizeMatrix; ++i ){
            count[i] = chunk / sizeMatrix;
        }
        count[commSize - 1] += sizeMatrix % commSize;
        int* shift = (int*)malloc(commSize * sizeof(int));
        shift[0] = 0;
        for(int i = 1; i < commSize; ++i ){
            shift[i] = count[i-1]+shift[i-1];
        }
        MPI_Gatherv(z0, chunk / sizeMatrix, MPI_DOUBLE, mainZ, count, shift, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(mainZ, sizeMatrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double* matrixPow = powMatrixCol(matrix, mainZ, sizeMatrix, chunk, 0);
        double* r1 = minusColumn(r0, powConfCol(alpha1, matrixPow, shortSize), shortSize, 0);
        betta1 = getBetta(r1, r0, shortSize);
        double* z1 = plusColumn(r1, powConfCol(betta1, z0, shortSize), shortSize, 0);
        free(x);
        free(r0);
        free(z0);
        x = NULL;
        r0 = NULL;
        z0 = NULL;
        x = x1;
        r0 = r1;
        z0 = z1;
        alpha1 = getAlpha(r0, matrix, z0, sizeMatrix, chunk);
        count++;
    }
    free(b);
    free(r0);
    free(z0);
    return x;
}

void printSol(double *sol, int shortSize){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    printf("cur rank %d  %d\n", rank, shortSize);
    for (int i = 0; i < size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == i)
        {
            for (int j = 0; j < shortSize; ++j)
            {
                printf("%lf\n", sol[j]);
            }
        }
    }
}

int main(int argc, char *argv[]){
    clock_t startTime, endTime;
    startTime = clock();
    int commRank, commSize;
    int sizeMatrix = 3;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    int sizeBigMatrix = sizeMatrix * sizeMatrix;
    int chunk = defineChunk(sizeBigMatrix);
    int standartStrFromChunk = returnstandartStrFromChunk(sizeBigMatrix);
    double *bigMatrix = initMatrix(sizeMatrix, chunk, standartStrFromChunk);
    double *b = initB(sizeBigMatrix);
    double *sol = initX(sizeBigMatrix);    
    sol = systemSolution(bigMatrix, b, sol, sizeBigMatrix, chunk, standartStrFromChunk);
    printSol(sol, chunk / sizeBigMatrix);
    free(bigMatrix);
    free(sol);
    endTime = clock();
    float workingTime = (float)(endTime - startTime) / CLOCKS_PER_SEC;
    printf(" %d working time %f \n", commRank, workingTime);
    MPI_Finalize();
    return 0;
}
