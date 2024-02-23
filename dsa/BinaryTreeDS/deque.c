#include<stdio.h>
#include<stdlib.h>
#define iter(var, start, stop, step) for(size_t var = start; var < stop; var += step)

typedef signed long int i64;
typedef unsigned long int u64;

typedef struct Node{
    u64 data;
    struct Node *prev;
    struct Node *next;
}Node;

typedef struct Deque{
    Node *head;
    Node *tail;
    u64 len;
}Deque;

Node* node_create(u64 data){
    Node *node = malloc(sizeof(Node));
    node->data = data;
    node->next = NULL;
    node->prev = NULL;
    return node;
}

void node_print(const Node *node){
    if(node == NULL)
        printf("NULL\n");
    else
        printf("Node Contains: %d\n", node->data);
}

Deque deque_init(){
    Deque deque;
    deque.head = NULL;
    deque.tail = NULL;
    deque.len = 0;
    return deque;
}

void deque_push(Deque *deque, u64 data){
    Node *node = node_create(data);

    if(deque->head == NULL && deque->tail == NULL){
        deque->head = node;
        deque->tail = node;
    }
    else{
        node->prev = deque->tail;
        deque->tail = node;
    }
    deque->len++;
}

u64 deque_pop(Deque *deque){
    if (deque->tail == NULL && deque->head == NULL && deque->len == 0)
        //If deque is empty, do nothing, return error value
        return __UINT64_MAX__;

    Node *node = deque->tail;
    u64 node_data = node->data;
    if (deque->tail == deque->head){
        //If deque has only one element, remove it
        deque->tail = NULL;
        deque->head = NULL;
        deque->len = 0;
    }
    else{
        deque->tail = node->prev;
        deque->len = deque->len - 1;
    }
    free(node);
    return node_data;
}

void deque_print(Deque *deque){
    Node *node = deque->head;
    while(node != NULL){
        printf("%llu ", node->data);
        node = node->next;
    }
    printf("\n");
}


int main(){
    Deque deque = deque_init();
    deque_push(&deque, 7);
    deque_push(&deque, 8);
    deque_push(&deque, 9);
    //deque_print(&deque);

    printf("%llu", deque.tail->next->data);

    //printf("Tail: %d, Stack Height: %d\n", deque.tail->data, deque.len);
    //printf("Popped Value: %d, Tail: %d, Stack Height: %d\n", 
            //deque_pop(&deque), deque.tail->data, deque.len);
    
}