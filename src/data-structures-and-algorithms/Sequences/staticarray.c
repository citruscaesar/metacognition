#include <stdio.h> 
#include <stdint.h>

void readArray(int64_t* array, size_t len);
void printArray(int64_t* array, size_t len);
void readMatrix(int64_t* matrix, size_t nrows, size_t ncols);
void printMatrix(int64_t* matrix, size_t nrows, size_t ncols);

int main(){
    size_t nrows, ncols;
    //printf("Enter Matrix Rows and Columns: ");
    //scanf("%d", &nrows);
    //scanf("%d", &ncols);
    nrows = 2;
    ncols = 2;

    int64_t matrix[nrows][ncols];
    readMatrix(matrix, nrows, ncols);

    printf("[");
    for(size_t r = 0; r < nrows; ++r){
        for(size_t c = 0; c < ncols; ++c){
            printf("%2d", matrix[r][c]);
            if (c != ncols-1)
                printf(" ");
        }
        if (r != nrows-1){printf("\n ");}
    }
    printf("]");

    return 0;
}

void printArray(int64_t* array, size_t len){
    printf("[");
    for(size_t i = 0; i<len; ++i){
        printf("%d", array[i]);
        if (i != len-1)
            printf(" ");
    }
    printf("]");
}

void readArray(int64_t* array, size_t len){
    for(size_t i = 0; i<len; ++i)
        scanf("%d", &array[i]);
}

void readMatrix(int64_t** matrix, size_t nrows, size_t ncols){
    for(size_t r = 0; r < nrows; ++r){
        for(size_t c = 0; c < ncols; ++c){
            //printf("(%d, %d)\n", r, c);
            scanf("%d", &matrix[r][c]);
        }
    }
}

//void printMatrix(int64_t* matrix, size_t nrows, size_t ncols){
    //printf("[");
    //for(size_t r = 0; r < nrows; ++r){
        //printf("[");
        //for(size_t c = 0; c < ncols; ++c){
            //printf("%d", matrix[r][c]);
            //if (c != ncols-1)
                //printf(" ");
        //}
        //printf("]");
    //}
    //printf("]");
//}