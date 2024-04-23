#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


#define EPSILON 0.00000001


double* phi;
double* oldPhi;
double* partOfPhi;
double* partOfOldPhi;

typedef struct TSize
{
    double Dx;
    double Dy;
    double Dz;
} TSize;

typedef struct TNet
{
    size_t Nx;
    size_t Ny;
    size_t Nz;
} TNet;

typedef struct TCoords
{
    int x;
    int y;
    int z;
} TCoords;

typedef struct TStep
{
    double hx;
    double hy;
    double hz;
} TStep;

double GetA(){
    return 100000;
}

int GetCount(int num, int count, int size){
    int startIndex = size * num / count;
    int endIndex = size * (num + 1) / count;
    int countLay = endIndex - startIndex;
    return countLay;
}

void InitSize(TSize* size){
    size->Dx = 2;
    size->Dy = 2;
    size->Dz = 2;
}

void InitNet(TNet* net){
    net->Nx = 500;
    net->Ny = 500;
    net->Nz = 500;
}

void InitStep(TStep* step, TSize* size, TNet* net ){
    step->hx = size->Dx/(net->Nx - 2);
    step->hy = size->Dy/(net->Ny - 2);
    step->hz = size->Dz/(net->Nz - 2);
}

void InitCoords(TCoords* coords, double x, double y, double z){
    coords->x = x;
    coords->y = y;
    coords->z = z;
}

void InitPhi(double * phi, TNet* net){
    for(size_t z = 1; z < net->Nz - 1; z++){
        for(size_t y = 1; y < net->Ny - 1; y++){
            for(size_t x = 1; x < net->Nx - 1; x++){
                int index = x + y * net->Nx + z * net->Ny * net->Nx;
                phi[index] = 0;
            }
        }
    }
}

double GetPhi(double x, double y, double z){
    double phi = x * x + y * y + z * z;
    return phi;
}

double GetP(TNet* net, double x, double y, double z, double a){
    int index =  x + y * net->Nx + z * net->Ny * net->Nx;;
    double p = 6 - a * GetPhi(x, y, z);
    return p;
}

void BorderConditions(TNet* net, TStep* step,TCoords* startCoords){
    double Z0 = startCoords->z + 0 * step->hz;
    double ZMax = startCoords->z +  (net->Nz - 1) * step->hz;
    for(size_t y = 0;  y < net-> Ny; ++y){
        for(size_t x = 0;  x < net-> Nx; ++x){
            int indexZ0  = x + y * net->Nx + 0 * net->Ny * net->Nx;
            int indexZMax  = x + y * net->Nx + (net->Nz - 1) * net->Ny * net->Nx;
            double X = startCoords->x + x * step->hx;
            double Y = startCoords->y + y * step->hy;
            phi[indexZ0] = GetPhi(X, Y, Z0);
            phi[indexZMax] = GetPhi(X, Y, ZMax);
        }
    }
    double Y0 = startCoords->y + 0 * step->hy;
    double YMax = startCoords->y +  (net->Ny - 1) * step->hy;
    for(size_t z = 0;  z < net-> Nz; ++z){
        for(size_t x = 0;  x < net-> Nx; ++x){
            int indexY0  = x + 0 * net->Nx + z * net->Ny * net->Nx;
            int indexYMax  = x + (net->Ny - 1) * net->Nx + z * net->Ny * net->Nx;
            double X = startCoords->x + x * step->hx;
            double Z = startCoords->z + z * step->hz;
            phi[indexY0] = GetPhi(X, Y0, Z);
            phi[indexYMax] = GetPhi(X, YMax, Z);
        }
    }
    double X0 = startCoords->x + 0 * step->hx;
    double XMax = startCoords->x +  (net->Nx - 1) * step->hx;
    for(size_t z = 0;  z < net-> Nz; ++z){
        for(size_t y = 0;  y < net-> Ny; ++y){
            int indexX0  = 0 + y * net->Nx + z * net->Ny * net->Nx;
            int indexXMax  = (net->Nx -1) + y * net->Nx + z * net->Ny * net->Nx;
            double Y = startCoords->y + y * step->hy;
            double Z = startCoords->z + z * step->hz;
            phi[indexX0] = GetPhi(X0, Y, Z);
            phi[indexXMax] = GetPhi(XMax, Y, Z);
        }
    }
}

double GetConf(TStep* step, double A){
    double stepX = step->hx * step->hx;
    double stepY = step->hy * step->hy;
    double stepZ = step->hz * step->hz;
    double denominator = 2/stepX + 2/stepY + 2/stepZ + A;
    double conf = 1/  denominator;
    return conf;
}

void CalculatePhi(TNet* net, TStep* step, TCoords* startCoords, int countLays, double conf, double a){
    int rankProc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankProc);
    int sizeProc;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeProc);
    double* leftLay = NULL;
    double* rightLay = NULL;
    MPI_Request requestRecvLeft = MPI_REQUEST_NULL;
    MPI_Request requestSendLeft = MPI_REQUEST_NULL;
    MPI_Request requestRecvRight = MPI_REQUEST_NULL;
    MPI_Request requestSendRight = MPI_REQUEST_NULL;
    int countInLay = net->Nx * net->Ny;
    if(rankProc != 0){
        MPI_Isend(partOfOldPhi, countInLay, MPI_DOUBLE, rankProc - 1, 0, MPI_COMM_WORLD, &requestSendLeft);
        leftLay = (double*)malloc(countInLay * sizeof(double));
        MPI_Irecv(leftLay, countInLay, MPI_DOUBLE, rankProc - 1, 0, MPI_COMM_WORLD, &requestRecvLeft);
    }
    if(rankProc != sizeProc - 1){
        int offset = (countLays - 1) * countInLay;
        MPI_Isend((partOfOldPhi + offset), countInLay, MPI_DOUBLE, rankProc + 1, 0, MPI_COMM_WORLD, &requestSendRight);
        rightLay = (double*)malloc(countInLay * sizeof(double));
        MPI_Irecv(rightLay, countInLay, MPI_DOUBLE, rankProc + 1, 0, MPI_COMM_WORLD, &requestRecvRight);
    }
    int startIndex  = net->Nz * rankProc / sizeProc;
    for(int k = startIndex + 1; k < startIndex + countLays - 1; ++k){
        for(int j = 1; j < net->Ny - 1; ++j){
            for(int i = 1; i < net->Nx - 1; ++i){
                int kIndex = k - startIndex;
                int mainIndex = i + j * net->Nx + kIndex * net->Ny * net->Nx;
                int lessIndexI = (i - 1) + j * net->Nx + kIndex * net->Ny * net->Nx;
                int bigIndexI = (i + 1) + j * net->Nx + kIndex * net->Ny * net->Nx;
                int lessIndexJ = i + (j - 1) * net->Nx + kIndex * net->Ny * net->Nx;
                int bigIndexJ = i + (j + 1) * net->Nx + kIndex * net->Ny * net->Nx;
                int lessIndexK = i + j * net->Nx + (kIndex - 1) * net->Ny * net->Nx;
                int bigIndexK = i + j * net->Nx + (kIndex + 1) * net->Ny * net->Nx;
                double xSum = (partOfOldPhi[bigIndexI] - partOfOldPhi[lessIndexI])/2;
                double ySum = (partOfOldPhi[bigIndexJ] - partOfOldPhi[lessIndexJ])/2;
                double zSum = (partOfOldPhi[bigIndexK] - partOfOldPhi[lessIndexK])/2;
                double stepX = step->hx * step->hx;
                double stepY = step->hy * step->hy;
                double stepZ = step->hz * step->hz;
                double Xi = startCoords->x + i *step->hx;
                double Yj = startCoords->x + i *step->hx;
                double Zk = startCoords->x + i *step->hx;
                double p = GetP(net, Xi, Yj, Zk, a);
                partOfPhi[mainIndex] = conf * (xSum/stepX + ySum/stepY + zSum/stepZ - p) ;
            }
        }
    }
    MPI_Wait(&requestRecvLeft, MPI_STATUS_IGNORE);
    if(rankProc != 0){
        int k = startIndex;
        for(int j = 1; j < net->Ny - 1; ++j){
            for(int i = 1; i < net->Nx - 1; ++i){
                int kIndex = k - startIndex;
                int mainIndex = i + j * net->Nx + kIndex * net->Ny * net->Nx;
                int lessIndexI = (i - 1) + j * net->Nx + kIndex * net->Ny * net->Nx;
                int bigIndexI = (i + 1) + j * net->Nx + kIndex * net->Ny * net->Nx;
                int lessIndexJ = i + (j - 1) * net->Nx + kIndex * net->Ny * net->Nx;
                int bigIndexJ = i + (j + 1) * net->Nx + kIndex * net->Ny * net->Nx;
                int lessIndexK = i + j * net->Nx;
                int bigIndexK = i + j * net->Nx + (kIndex + 1) * net->Ny * net->Nx;
                double xSum = (partOfOldPhi[bigIndexI] - partOfOldPhi[lessIndexI])/2;
                double ySum = (partOfOldPhi[bigIndexJ] - partOfOldPhi[lessIndexJ])/2;
                double zSum = (partOfOldPhi[bigIndexK] - leftLay[lessIndexK])/2;
                double stepX = step->hx * step->hx;
                double stepY = step->hy * step->hy;
                double stepZ = step->hz * step->hz;
                double Xi = startCoords->x + i *step->hx;
                double Yj = startCoords->x + i *step->hx;
                double Zk = startCoords->x + i *step->hx;
                double p = GetP(net, Xi, Yj, Zk, a);
                partOfPhi[mainIndex] = conf * (xSum/stepX + ySum/stepY + zSum/stepZ - p);
            }
        }
    }
    MPI_Wait(&requestRecvRight, MPI_STATUS_IGNORE);
    if(rankProc != sizeProc - 1){
        int k = startIndex + countLays - 1;
        for(int j = 1; j < net->Ny - 1; ++j){
            for(int i = 1; i < net->Nx - 1; ++i){
                int kIndex = k - startIndex;
                int mainIndex = i + j * net->Nx + kIndex  * net->Ny * net->Nx;
                int lessIndexI = (i - 1) + j * net->Nx +kIndex  * net->Ny * net->Nx;
                int bigIndexI = (i + 1) + j * net->Nx + kIndex  * net->Ny * net->Nx;
                int lessIndexJ = i + (j - 1) * net->Nx +  kIndex  * net->Ny * net->Nx;
                int bigIndexJ = i + (j + 1) * net->Nx + kIndex  * net->Ny * net->Nx;
                int lessIndexK = i + j * net->Nx + (kIndex  - 1) * net->Ny * net->Nx;
                int bigIndexK = i + j * net->Nx;
                double xSum = (partOfOldPhi[bigIndexI] - partOfOldPhi[lessIndexI])/2;
                double ySum = (partOfOldPhi[bigIndexJ] - partOfOldPhi[lessIndexJ])/2;
                double zSum = (rightLay[bigIndexK] - partOfOldPhi[lessIndexK])/2;
                double stepX = step->hx * step->hx;
                double stepY = step->hy * step->hy;
                double stepZ = step->hz * step->hz;
                double Xi = startCoords->x + i *step->hx;
                double Yj = startCoords->x + i *step->hx;
                double Zk = startCoords->x + i *step->hx;
                double p = GetP(net, Xi, Yj, Zk, a);
                partOfPhi[mainIndex] = conf * (xSum/stepX + ySum/stepY + zSum/stepZ - p);
            }
        }
    }
    free(leftLay);
    free(rightLay);
}

double Max(double* array, TNet* net, int countLays){
    double max = -__DBL_MAX__;
    int countInLay = net->Nx * net->Ny;
    int count = countLays * countInLay;
    int findIndex = 0;
    for(int k =  1; k < countLays - 1; ++k) {
        for (int j = 1; j < net->Ny - 1; ++j) {
            for (int i = 1; i < net->Nx - 1; ++i) {
                int mainIndex = i + j * net->Nx + k * net->Nx * net->Ny;
                double curVal = array[mainIndex];
                if (curVal > max) {
                    findIndex = mainIndex;
                    max = curVal;
                }
            }
        }
    }
    return max;
}

char NotGoodApproximation(TNet* net, int countLay){
    MPI_Barrier(MPI_COMM_WORLD);
    int rankProc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankProc);
    int sizeProc;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeProc);
    int countElem = GetCount(rankProc,sizeProc, net->Nz) * net->Nx * net->Ny;
    char mainResult = 1;
    double maxPhi = Max(partOfPhi, net, countLay);
    double maxOldPhi = Max(partOfOldPhi, net, countLay);
    double* maxAllPhi  = (double*) malloc(sizeProc * sizeof(double));
    double* maxAllOldPhi  = (double*) malloc(sizeProc * sizeof(double));
    MPI_Gather(&maxPhi, 1, MPI_DOUBLE, maxAllPhi , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&maxOldPhi, 1, MPI_DOUBLE, maxAllOldPhi , 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rankProc == 0){
        double realMaxPhi = maxAllPhi[0];
        for(int i = 1; i < sizeProc; ++i){
            if(maxAllPhi[i] > realMaxPhi){
                realMaxPhi = maxAllPhi[i] ;
            }
        }
        double realMaxOldPhi = maxAllOldPhi[0];
        for(int i = 1; i < sizeProc; ++i){
            if(maxAllOldPhi[i] > realMaxOldPhi){
                realMaxOldPhi = maxAllOldPhi[i] ;
            }
        }
        double realResult = realMaxPhi - realMaxOldPhi;
        if(realResult < 0 ){
            realResult *= -1;
        }
        if(realResult < EPSILON){
            mainResult = 0;
        }
    }
    MPI_Bcast(&mainResult, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    free(maxAllOldPhi);
    free(maxAllPhi);
    return mainResult;
}

void Solution(TNet* net, TStep* step, TCoords* startCoords, int countLay, double a, int netSize){
    double conf = GetConf(step, a);
    CalculatePhi(net, step, startCoords,countLay, conf, a);
    while(NotGoodApproximation(net, countLay)){
        memcpy(partOfOldPhi, partOfPhi, netSize * sizeof(double));
        CalculatePhi(net, step, startCoords, countLay, conf, a);
    }
}

void PrintResult(TNet* net){
    for(int k = 0; k < net->Nz; ++k){
        for(int j = 0; j < net->Ny; ++j){
            for(int i = 0; i < net->Nx; ++i){
                int mainIndex = i + j * net->Nx + k * net->Ny * net->Nx;
                printf("phi (%d %d %d) : %lf \n", i, j, k, phi[mainIndex]);
            }
        }
    }
}

void SharedPhi(TNet* net, int sizeProc, int rankProc){
    int* counts = (int*)malloc(sizeProc * sizeof(int));
    int countLays = net->Nz;
    int countInLay = net->Ny * net->Nx;
    for(int i = 0; i < sizeProc; ++i){
        counts[i] = GetCount(i, sizeProc, countLays) * countInLay;
    }
    int* shift = (int*)malloc(sizeProc * sizeof(int));
    shift[0] = 0;
    for(int i = 1; i < sizeProc; ++i){
        shift[i] = shift[i - 1] + counts[i - 1];
    }
    MPI_Scatterv(phi, counts, shift, MPI_DOUBLE, partOfPhi,  counts[rankProc], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    free(shift);
    free(counts);
}

void UnionParts(TNet* net, int sizeProc, int rankProc){
    int* counts = (int*)malloc(sizeProc * sizeof(int));
    int countLays = net->Nz;
    int countInLay = net->Ny * net->Nx;
    for(int i = 0; i < sizeProc; ++i){
        counts[i] = GetCount(i, sizeProc, countLays) * countInLay;
    }
    int* shift = (int*)malloc(sizeProc * sizeof(int));
    shift[0] = 0;
    for(int i = 1; i < sizeProc; ++i){
        shift[i] = shift[i - 1] + counts[i - 1];
    }
    MPI_Gatherv(partOfPhi, counts[rankProc], MPI_DOUBLE, phi, counts, shift, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(shift);
    free(counts);
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    double start_time, end_time;
    start_time = MPI_Wtime();
    int rankProc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankProc);
    int sizeProc;
    MPI_Comm_size(MPI_COMM_WORLD, &sizeProc);
    double a = GetA();
    TSize size;
    InitSize(&size);
    TNet net;
    InitNet(&net);
    TCoords startCoords;
    InitCoords(&startCoords, -1, -1, -1);
    TStep step;
    InitStep(&step, &size, &net);
    size_t netSize = net.Nx * net.Ny * net.Nz;
    if(rankProc == 0){
        phi = (double*)malloc(netSize * sizeof(double));
        InitPhi(phi, &net);
        BorderConditions(&net, &step, &startCoords);
    }
    else{
        oldPhi = NULL;
        phi = NULL;
    }
    int countLay  = GetCount(rankProc, sizeProc, net.Nz);
    int countElem = countLay * net.Nx * net.Ny;
    partOfPhi = (double*)malloc(countElem * sizeof(double));
    partOfOldPhi = (double*)malloc(countElem * sizeof(double));
    SharedPhi(&net, sizeProc, rankProc);
    int localNetSize = countLay * net.Ny * net.Nz;
    memcpy(partOfOldPhi, partOfPhi, localNetSize * sizeof(double));
    Solution(&net, &step, &startCoords, countLay, a, localNetSize);
    UnionParts(&net, sizeProc, rankProc);
    if(rankProc == 0){
        //PrintResult(&net);
    }
    free(partOfOldPhi);
    free(partOfPhi);
    free(phi);
    free(oldPhi);
    end_time = MPI_Wtime();
    if(rankProc == 0){
        printf("Total time: %f seconds\n", end_time - start_time);
    }
    MPI_Finalize();
    return 0;
}
