#include <stdio.h> 
#include <stdlib.h>
typedef long long unsigned int u64;

typedef struct Node{u64 data; struct Node* next; struct Node* prev;} Node;
typedef struct Deque{struct Node* head; struct Node* tail; u64 length;} Deque;

Node* create_node(u64 data){
    Node* node = (Node*)malloc(sizeof(Node));
    node->data = data;
    return node;
}

void push_last(Deque* deque, u64 data);
void push_first(Deque* deque, u64 data);
u64 pop_last(Deque* deque);
u64 pop_first(Deque* deque);
u64 len(Deque* deque);
u64 display(Deque* deque);

int main(){

    return 0;
}