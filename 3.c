#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

struct TColors{
    int colorAll;
    int colorB; 
    int colorA;
    int colorS;
};

void InitA(int* A, int size){
    for(int i = 0; i < size; ++i){
        A[i] = i;
    }
}

void InitB(int* B, int size){
    for(int i = 0; i < size; ++i){
        B[i] = i + 1;
    }
}

void PrintC(int* C, int sizeX, int sizeY){
    for(int i = 0; i < sizeY; ++i){
        for(int j = 0; j < sizeX; ++j){
            printf("%d ", C[i * sizeY + j]);
        }
        printf("\n");
    }

}

void MultMatrix(int* C, int* A, int* B, int sizeAY, int sizeBX, int genSize){
    for(int i = 0; i < sizeAY; ++i){
        for(int j = 0; j < sizeBX; ++j){
            for(int k = 0; k < genSize; ++k){
                C[i * sizeBX + j ] += A[genSize * i + k] * B[j * genSize + k];                
            }
        }
   }
}

int GetCount(int num, int count, int size){
    int startIndex = size * num / count;
    int endIndex = size * (num + 1) / count;
    int countRow = endIndex - startIndex;
    return countRow;
}

int GetCountColls(int num, int count, int size){
    int startIndex = size * num / count;
    int endIndex = size* (num + 1) / count;
    int countRow = endIndex - startIndex;
    return countRow;
}

int  InitMatrixs(int** A, int** B, int sizeAX, int sizeAY, int sizeBX, int sizeBY, int MyX, int MyY){
    if(MyX == 0 && MyY == 0){
        if(sizeAX != sizeBY){
            printf("wrong size matrix\n");
            return 1;
        }
        *A = (int*)malloc(sizeAX * sizeAY * sizeof(int));
        InitA(*A, sizeAX * sizeAY);
        *B = (int*)malloc(sizeBX * sizeBY * sizeof(int));
        InitB(*B, sizeAX * sizeAY);
    }
    return 0;
}

void ShareY0(MPI_Comm* COMM_Y0, int** localB, int* B, int MyX, int MyY,int sizeBX,  int sizeBY, int X){
    MPI_Datatype MPI_Column;
    MPI_Type_vector(sizeBY, 1, sizeBX, MPI_INT, &MPI_Column);
    MPI_Type_commit(&MPI_Column);
    MPI_Datatype MPI_Column_offset;
    MPI_Type_create_resized(MPI_Column, 0, sizeof(int), &MPI_Column_offset);
    MPI_Type_commit(&MPI_Column_offset);
    int color = MyY == 0 ? 1: 0;
    int key = MyX;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, COMM_Y0);
    int myCountElem = GetCount(MyX, X, sizeBY) * sizeBY;
    *localB = (int*) malloc(myCountElem * sizeof(int));
    int* counts = (int*)malloc(X * sizeof(int));
    int* shift = (int*)malloc(X * sizeof(int));
    for(int i = 0; i < X; ++i){
        counts[i] = GetCount(i, X, sizeBY);
    }
    shift[0] = 0;
    for(int i = 1; i < X; ++i){
        shift[i] = shift[i - 1] + counts[i - 1];
    } 
    if(color == 1){
        MPI_Scatterv(B, counts, shift, MPI_Column_offset, *localB, myCountElem, MPI_INT,  0, *COMM_Y0);
    }
    free(shift);
    free(counts);
    MPI_Type_free(&MPI_Column);
    MPI_Type_free(&MPI_Column_offset);
    MPI_Barrier(MPI_COMM_WORLD);
}

    void ShareX0(MPI_Comm* COMM_X0, int** localA, int* A, int MyX, int MyY, int sizeAX, int Y){
        int color = MyX == 0 ? 1 : 0;
        int key = MyY;
        MPI_Comm_split(MPI_COMM_WORLD, color, key, COMM_X0);
        int rankX0, sizeX0;
        MPI_Comm_rank(*COMM_X0, &rankX0);
        MPI_Comm_size(*COMM_X0, &sizeX0);
        int MyCountElem = GetCount(MyY, Y, sizeAX ) * sizeAX;
        *localA = (int*) malloc(sizeof(int) * MyCountElem);
        int* counts = (int*)malloc(Y * sizeof(int));
        for(int i = 0; i < Y; ++i){
            counts[i] = GetCount(i, Y, sizeAX) * sizeAX;
        }
        int* shift = (int*)malloc(Y * sizeof(int));
        shift[0] = 0;
        for(int i = 1; i < Y; ++i){
            shift[i] = shift[i - 1] + counts[i - 1];
        }
        if(color == 1){
            MPI_Scatterv(A, counts, shift, MPI_INT, *localA,  counts[MyY], MPI_INT, 0, *COMM_X0);
        }
        free(shift);
        free(counts);
        MPI_Barrier(MPI_COMM_WORLD);
    }

void ShareLocalA(int* localA, int MyX, int MyY, int X, int Y, int sizeAX){
    MPI_Comm* lineComms = (MPI_Comm*)malloc(Y * sizeof(MPI_Comm));
    for(int i = 0; i < Y; ++i){
        int colorLine = MyY == i ? 1 : 0;
        int keyLine = MyX;
        MPI_Comm_split(MPI_COMM_WORLD, colorLine, keyLine, &lineComms[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int MyCountElem = GetCount(MyY, Y, sizeAX ) * sizeAX;
    MPI_Bcast(localA, MyCountElem, MPI_INT, 0, lineComms[MyY]);
    for(int i = 0; i < Y; ++i){
        MPI_Comm_free(&lineComms[i]);
    }
    free(lineComms);
    MPI_Barrier(MPI_COMM_WORLD);
}

void ShareLocalB(int* localB, int MyX, int MyY, int X, int Y, int sizeBY){
    MPI_Comm* colComms = (MPI_Comm*)malloc(X* sizeof(MPI_Comm));
    for(int i = 0; i < X; ++i){
        int colorCol = MyX == i ? 1 : 0;
        int keyCol = MyY;
        MPI_Comm_split(MPI_COMM_WORLD, colorCol, keyCol, &colComms[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int MyCountElem = GetCount(MyX, X, sizeBY ) * sizeBY;
    MPI_Bcast(localB, MyCountElem, MPI_INT, 0, colComms[MyX]);
    for(int i = 0; i < X; ++i){
        MPI_Comm_free(&colComms[i]);
    }
    free(colComms);
    MPI_Barrier(MPI_COMM_WORLD);
}

void DefineNodeType(struct TColors* colors, int X, int Y, int sizeAX, int sizeBY, int localACount, int localBCount, int rank){
    int standartCountA = GetCount(0, Y, sizeAX); 
    int standartCountB = GetCount(0, X, sizeBY);
    int bigCountA = standartCountA + 1; 
    int bigCountB = standartCountB + 1; 
    int bBig = localBCount == bigCountB ? 1: 0;
    int aBig = localACount == bigCountA ? 1: 0;
    colors->colorAll = 0;
    colors->colorB = 0; 
    colors->colorA = 0;
    colors->colorS = 0;
    if(rank == 0){
        colors->colorAll = 1;
        colors->colorB = 1;
        colors->colorA = 1;
        colors->colorS = 1;
    }
    if(bBig  && aBig){
        colors->colorAll = 1;
    }
    else if (bBig){
        colors->colorB = 1;
    }
    else if (aBig){
        colors->colorA = 1;
    }
    else{
        colors->colorS = 1;
    }
}

void DistributeNodes(MPI_Comm* COMM_BIG_ALL, MPI_Comm* COMM_BIG_B, MPI_Comm* COMM_BIG_A, MPI_Comm* COMM_STANDART, struct TColors* colors, int rank){
    MPI_Comm_split(MPI_COMM_WORLD, colors->colorAll, rank, COMM_BIG_ALL);
    MPI_Comm_split(MPI_COMM_WORLD, colors->colorB, rank, COMM_BIG_B);
    MPI_Comm_split(MPI_COMM_WORLD, colors->colorA, rank, COMM_BIG_A);
    MPI_Comm_split(MPI_COMM_WORLD, colors->colorS, rank, COMM_STANDART);
}

void PrintResult(int* result, int sizeX, int sizeY){
    for(int i = 0; i < sizeY; ++i ){
        for(int j = 0; j < sizeX; ++j){
            printf("%d ", result[i * sizeX + j]);
        }
        printf("\n");
    }
}

void ReturnBigB(MPI_Comm COMM_BIG_B, struct  TColors* colors, int* C, int* localC, int X, int Y, int sizeAX, int sizeBY, int localACount, int localBCount, int localCCount){
    MPI_Datatype MPI_BigB;
    MPI_Type_vector(localACount, localBCount + 1, sizeAX, MPI_INT, &MPI_BigB);
    MPI_Type_commit(&MPI_BigB);
    MPI_Datatype MPI_BigB_offset;
    MPI_Type_create_resized(MPI_BigB, 0, sizeof(int), &MPI_BigB_offset);
    MPI_Type_commit(&MPI_BigB_offset);
    if (colors->colorB == 1){
        int commBSize;
        MPI_Comm_size(COMM_BIG_B, &commBSize);
        int* recvCounts = (int*)malloc(commBSize * sizeof(int));
        recvCounts[0] = 0;
        for(int i = 1; i < commBSize; ++i){
            recvCounts[i] = 1;
        }
        int* displs = (int*)malloc(commBSize * sizeof(int));
        displs[0] = 0;
        int curIndex = 1;
        for(int i = 0; i < Y; ++i){
            for(int j = 0; j < X; ++j){
                int lACount = GetCount(i, Y, sizeAX);
                int lBCount = GetCount(j, X, sizeBY);
                if(lBCount == GetCount(0, X, sizeBY) + 1 && lACount == GetCount(0, Y, sizeAX) ){
                    int startIndexX = sizeAX * j / X;
                    int startIndexY = sizeBY * i / Y;
                    displs[curIndex] = sizeAX * startIndexY + startIndexX;
                    curIndex++;
                }
            }

        }
        
        MPI_Gatherv(localC, localCCount, MPI_INT, C, recvCounts, displs, MPI_BigB_offset, 0, COMM_BIG_B);
        free(recvCounts);
        free(displs);
    }
    MPI_Type_free(&MPI_BigB);
    MPI_Type_free(&MPI_BigB_offset);
}

void ReturnStandart(MPI_Comm COMM_STANDART, struct  TColors* colors, int* C, int* localC, int X, int Y, int sizeAX, int sizeBY, int localACount, int localBCount, int localCCount){
    MPI_Datatype MPI_Standart;
    MPI_Type_vector(localACount, localBCount, sizeAX, MPI_INT, &MPI_Standart);
    MPI_Type_commit(&MPI_Standart);
    MPI_Datatype MPI_Standart_offset;
    MPI_Type_create_resized(MPI_Standart, 0, sizeof(int), &MPI_Standart_offset);
    MPI_Type_commit(&MPI_Standart_offset);
    if (colors->colorS == 1){
        int commStandartSize;
        MPI_Comm_size(COMM_STANDART, &commStandartSize);
        int* recvCounts = (int*)malloc(commStandartSize * sizeof(int));
        for(int i = 0; i < commStandartSize; ++i){
            recvCounts[i] = 1;
        }
        int* displs = (int*)malloc(commStandartSize * sizeof(int));
        int curIndex = 0;
        for(int i = 0; i < Y; ++i){
            for(int j = 0; j < X; ++j){
                int lACount = GetCount(i, Y, sizeAX);
                int lBCount = GetCount(j, X, sizeBY);
                if(lBCount == GetCount(0, X, sizeBY) && lACount == GetCount(0, Y, sizeAX) ){
                    int startIndexX = sizeAX * j / X;
                    int startIndexY = sizeBY * i / Y;
                    displs[curIndex] = sizeAX * startIndexY + startIndexX;
                    curIndex++;
                }
            }

        }
        MPI_Gatherv(localC, localCCount, MPI_INT, C, recvCounts, displs, MPI_Standart_offset, 0, COMM_STANDART);
        free(recvCounts);
        free(displs);
    }
    MPI_Type_free(&MPI_Standart);
    MPI_Type_free(&MPI_Standart_offset);
}

void ReturnBigAll(MPI_Comm COMM_BIG_ALL, struct  TColors* colors, int* C, int* localC, int X, int Y, int sizeAX, int sizeBY, int localACount, int localBCount, int localCCount){
    MPI_Datatype MPI_BigALL;
    MPI_Type_vector(localACount + 1, localBCount + 1, sizeAX, MPI_INT, &MPI_BigALL);
    MPI_Type_commit(&MPI_BigALL);
    MPI_Datatype MPI_BigALL_offset;
    MPI_Type_create_resized(MPI_BigALL, 0, sizeof(int), &MPI_BigALL_offset);
    MPI_Type_commit(&MPI_BigALL_offset);
    if (colors->colorAll == 1){
        int commALLSize;
        MPI_Comm_size(COMM_BIG_ALL, &commALLSize);
        int* recvCounts = (int*)malloc(commALLSize * sizeof(int));
        recvCounts[0] = 0;
        for(int i = 1; i < commALLSize; ++i){
            recvCounts[i] = 1;
        }
        int* displs = (int*)malloc(commALLSize * sizeof(int));
        displs[0] = 0;
        int curIndex = 1;
        for(int i = 0; i < Y; ++i){
            for(int j = 0; j < X; ++j){
                int lACount = GetCount(i, Y, sizeAX);
                int lBCount = GetCount(j, X, sizeBY);
                if(lBCount == GetCount(0, X, sizeBY) + 1 && lACount == GetCount(0, Y, sizeAX) + 1){
                    int startIndexX = sizeAX * j / X;
                    int startIndexY = sizeBY * i / Y;
                    displs[curIndex] = sizeAX * startIndexY + startIndexX;
                    curIndex++;
                }
            }

        }
        
        MPI_Gatherv(localC, localCCount, MPI_INT, C, recvCounts, displs, MPI_BigALL_offset, 0, COMM_BIG_ALL);
        free(recvCounts);
        free(displs);
    }
    MPI_Type_free(&MPI_BigALL);
    MPI_Type_free(&MPI_BigALL_offset);
}

void ReturnBigA(MPI_Comm COMM_BIG_A, struct  TColors* colors, int* C, int* localC, int X, int Y, int sizeAX, int sizeBY, int localACount, int localBCount, int localCCount){
    MPI_Datatype MPI_BigA;
    MPI_Type_vector(localACount + 1, localBCount, sizeAX, MPI_INT, &MPI_BigA);
    MPI_Type_commit(&MPI_BigA);
    MPI_Datatype MPI_BigA_offset;
    MPI_Type_create_resized(MPI_BigA, 0, sizeof(int), &MPI_BigA_offset);
    MPI_Type_commit(&MPI_BigA_offset);
    if (colors->colorA == 1){
        int commASize;
        MPI_Comm_size(COMM_BIG_A, &commASize);
        int* recvCounts = (int*)malloc(commASize * sizeof(int));
        recvCounts[0] = 0;
        for(int i = 1; i < commASize; ++i){
            recvCounts[i] = 1;
        }
        int* displs = (int*)malloc(commASize * sizeof(int));
        displs[0] = 0;
        int curIndex = 1;
        for(int i = 0; i < Y; ++i){
            for(int j = 0; j < X; ++j){
                int lACount = GetCount(i, Y, sizeAX);
                int lBCount = GetCount(j, X, sizeBY);
                if(lBCount == GetCount(0, X, sizeBY) && lACount == GetCount(0, Y, sizeAX) + 1 ){
                    int startIndexX = sizeAX * j / X;
                    int startIndexY = sizeBY * i / Y;
                    displs[curIndex] = sizeAX * startIndexY + startIndexX;
                    curIndex++;
                }
            }

        }
        
        MPI_Gatherv(localC, localCCount, MPI_INT, C, recvCounts, displs, MPI_BigA_offset, 0, COMM_BIG_A);
        free(recvCounts);
        free(displs);
    }
    MPI_Type_free(&MPI_BigA);
    MPI_Type_free(&MPI_BigA_offset);
}


int main(int argc, char *argv[]){
    int sizeAX = 2000;
    int sizeAY = 2000;
    int sizeBX = 2000;
    int sizeBY = 2000;
    MPI_Init(&argc, &argv);
    double start_time, end_time;
    start_time = MPI_Wtime();
    int X = 1;
    int Y = 1;
    int dims[2] = {X, Y};
    int periods[2] = {0, 0};
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm_2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm_2d);
    int coords[2];
    MPI_Cart_coords(comm_2d, rank, 2, coords);
    int MyX = coords[0];
    int MyY = coords[1];
    int* A = NULL;
    int* B = NULL;
    if (InitMatrixs(&A, &B, sizeAX, sizeAY, sizeBX,sizeBY, MyX, MyY) == 1){
        return 1;
    }
    MPI_Comm COMM_X0;
    int* localA = NULL;
    ShareX0(&COMM_X0, &localA, A, MyX, MyY, sizeAX, Y);
    ShareLocalA(localA, MyX, MyY, X, Y, sizeAX);
    int* localB = NULL;
    MPI_Comm COMM_Y0;
    ShareY0(&COMM_Y0, &localB, B, MyX, MyY, sizeBX, sizeBY, X);
    ShareLocalB(localB, MyX, MyY, X, Y, sizeBY);
    int localACount = GetCount(MyY, Y, sizeAX);
    int localBCount = GetCount(MyX, X, sizeBY);
    int localCCount = localACount * localBCount;
    int* localC = (int*)calloc(localCCount,  sizeof(int));
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MultMatrix(localC, localA, localB, localACount, localBCount, sizeAX);
    
    MPI_Comm COMM_STANDART;
    MPI_Comm COMM_BIG_A;
    MPI_Comm COMM_BIG_B;
    MPI_Comm COMM_BIG_ALL;
    struct TColors colors;
    DefineNodeType(&colors, X, Y, sizeAX, sizeBY,localACount,localBCount, rank);    
    DistributeNodes(&COMM_BIG_ALL, &COMM_BIG_B, &COMM_BIG_A, &COMM_STANDART, &colors, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    int* C = NULL;
    if(rank == 0){
        C = (int*)calloc(sizeAX * sizeBY, sizeof(int));
    }
    ReturnBigB(COMM_BIG_B, &colors, C, localC, X, Y, sizeAX, sizeBY, localACount, localBCount, localCCount);
    ReturnBigA(COMM_BIG_A, &colors, C, localC, X, Y, sizeAX, sizeBY, localACount, localBCount, localCCount);
    ReturnBigAll(COMM_BIG_ALL, &colors, C, localC, X, Y, sizeAX, sizeBY, localACount, localBCount, localCCount);
    ReturnStandart(COMM_STANDART, &colors, C, localC, X, Y, sizeAX, sizeBY, localACount, localBCount, localCCount);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    if(rank == 0){
        printf("Total time: %f seconds\n", end_time - start_time);
        //PrintResult(C, sizeAX, sizeBY);
    }
    free(localA);
    free(localB);
    free(localC);
    free(A);
    free(B);
    free(C);
    MPI_Comm_free(&COMM_STANDART);
    MPI_Comm_free(&COMM_BIG_A);
    MPI_Comm_free(&COMM_BIG_ALL);
    MPI_Comm_free(&COMM_BIG_B);
    MPI_Comm_free(&comm_2d);
    MPI_Comm_free(&COMM_X0);
    MPI_Comm_free(&COMM_Y0);
    MPI_Finalize();

    return 0;
}
