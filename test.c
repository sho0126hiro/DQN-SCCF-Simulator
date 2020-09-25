// # aの配列をソートするプログラムを書いてください
// # a = {3, 8, 6, 1, 4}

#include <stdio.h>
#define N 5

int main(){
  int a[N] = {3, 8, 6, 1, 4};
  int tmp, i, j;
  
  // sort
  for(i =0; i<N-1; i++){
    for(j=0; j<N-i; j++){
    	if(a[j] > a[j+1]){
        tmp = a[j+1];
        a[j+1]  = a[j];
        a[j] = tmp;
        }
    }
  }
    
  // output result
  for(i = 0; i<N; i++){
  	printf("%d ", a[i]);
  }
}