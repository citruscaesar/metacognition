#include<stdio.h>
#include<stdint.h>
#include<stdbool.h>
#include<stdlib.h>

typedef struct Node{
    int64_t data;
    struct Node *next;
    struct Node *prev;
} Node;

typedef struct Deque{
    struct Node *head;
    struct Node *tail;
    uint64_t len;
} Deque;

Node* init_node(int64_t data){
    Node *node = (Node *)malloc(sizeof(Node));
    node->data = data; node->next = NULL; node->prev = NULL;
    return node;
}
void step_forward(Node* node){
    node = node->next;
}

void step_backward(Node* node){
    node = node->prev;
}

Deque init_deque(){
    Deque deque = {NULL, NULL, 0};
    return deque;
}

bool is_empty(Deque *deque){
    return (deque->head == NULL && deque->tail == NULL && deque->len == 0) ? true : false;
}

void push_last(Deque *deque, int64_t data){
    Node *node = init_node(data);
    if (is_empty(deque)){
        deque->head = node; deque->tail = node;
    }
    else {
        node->prev = deque->tail; 
        deque->tail->next = node;
        deque->tail = node;
    }
    deque->len += 1;
}

void push_first(Deque *deque, int64_t data){
    Node *node = init_node(data);
    if (is_empty(deque)){
            deque->head = node; deque->tail = node;
    }
    else {
        node->next = deque->head;
        deque->head->prev = node;
        deque->head = node;
    }
    deque->len += 1;
}

int64_t pop_last(Deque* deque){
    if(is_empty(deque)) { return INT64_MIN; }
    else{
        int64_t pop_val = deque->tail->data;
        deque->tail = deque->tail->prev;
        free(deque->tail->next); deque->tail->next = NULL;
        return pop_val;
    }
}

int64_t pop_first(Deque* deque){
    if(is_empty(deque)) { return INT64_MIN; }
    else{
        int64_t pop_val = deque->head->data;
        deque->head = deque->head->next;
        free(deque->head->prev); deque->head->prev = NULL;
        return pop_val;
    }
}

void display(Deque *deque){
    Node *node = deque->head;
    printf("Deque(%llu): ", deque->len);
    while(node != NULL){
        printf("%lli ", node->data);
        node = node->next;
    }
    printf("\n");
}


int main(){
    Deque list = init_deque();
    push_last(&list, 69);
    push_first(&list, 12);
    push_last(&list, 420);
    display(&list);
    return 0;
}