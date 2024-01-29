#include <stdio.h> 
#include <stdlib.h>
#define iter(i, l) for(size_t i = 0; i < l; ++i)

typedef signed long int i64;

void readArray(i64*, size_t);
void printArray(i64*, size_t);
void readMatrix(i64**, size_t, size_t);
void printMatrix(i64**, size_t , size_t);
void deleteMatrix(i64**, size_t , size_t);

int main(){

    size_t nrows, ncols;
    //printf("Enter Matrix Rows and Columns: ");
    //scanf("%d", &nrows);
    //scanf("%d", &ncols);
    nrows = 2;
    ncols = 2;

    i64* matrix[nrows];

    // matrix = {i64* ptr_to_row1, i64* ptr_to_row2}
    // temp_row = {i64 num1, i64 num2, ...}
    // temp_row = &num1
    // ptr_to_row1 = temp_row = &num1 

    readMatrix(matrix, nrows, ncols);
    printMatrix(matrix, nrows, ncols);
}

void readArray(i64* array, size_t len){
    for(size_t i = 0; i<len; ++i)
        scanf("%lld", &array[i]);
}

void printArray(i64* array, size_t len){
    printf("[");
    for(size_t i = 0; i<len; ++i){
        printf("%2lld", array[i]);
        if (i != len-1) {printf(" ");}
    }
    printf("]");
}

void readMatrix(i64** matrix, size_t nrows, size_t ncols){
    for(size_t r = 0; r<nrows; r++){
        matrix[r] = (i64*)calloc(ncols, sizeof(i64));
        readArray(matrix[r], ncols);
    }
}

void printMatrix(i64** matrix, size_t nrows, size_t ncols){
    printf("[");
    for(size_t r = 0; r < nrows; ++r){
        printArray(matrix[r], ncols);
        if (r != nrows-1) {printf("\n ");}
    }
    printf("]");
}

void deleteMatrix(i64** matrix, size_t nrows, size_t ncols){
    for(size_t r = 0; r < nrows; ++r){
        free(matrix[r]);
    }
}