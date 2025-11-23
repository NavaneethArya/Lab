#include <stdio.h>
#include<omp.h>

int main(){
    int n;
    printf("enter number of iterations:");
    scanf("%d",&n);

    omp_set_num_threads(4);

    #pragma omp parallel for schedule(static,2)
    for(int i=0;i<n;i++){
        int tid = omp_get_thread_num();
        printf("Thread %d is executing iteration %d\n", tid, i);
    }
    return 0;
}


#include<stdio.h>
#include<omp.h>
#include<stdlib.h>

#define N 100000

void merge(int *a, int l, int m, int r) {
    int n1 = m - l + 1, n2 = r - m, i, j, k;
    int *L = malloc(n1 * sizeof(int));
    int *R = malloc(n2 * sizeof(int));

    for(i = 0; i < n1; i++) L[i] = a[l + i];
    for(j = 0; j < n2; j++) R[j] = a[m + 1 + j];

    i = j = 0;
    k = l;

    while(i < n1 && j < n2) {
        a[k++] = (L[i] < R[j]) ? L[i++] : R[j++];
    }
    while(i < n1)a[k++] = L[i++];
    while(j < n2)a[k++] = R[j++];

    free(L);
    free(R);
}

void sort(int *a, int l, int r, int d) {
    if(l>=r) return;
    int m = ( l + r ) / 2;

    if(d > 0) {
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("Thread %d working on left [%d..%d]\n", omp_get_thread_num(), l, m);
            sort(a, l, m, d-1);
        }
        #pragma omp section
        {
            printf("Thread %d working on right [%d..%d]\n", omp_get_thread_num(),m + 1, r);
            sort(a, m + 1, r, d-1);
        }
    }
    } else {
        sort(a, l, m, 0);
        sort(a, m + 1, r, 0);
    }
    merge(a, l, m, r);
}

int main() {
    omp_set_nested(1);

    int *a = malloc(N * sizeof(int));
    int *b = malloc(N * sizeof(int));

    for(int i = 0; i < N; i++)
        a[i] = b[i] = rand();

    double t = omp_get_wtime();
    sort(a, 0, N -1, 0);
    printf("Seq: %f seconds\n",omp_get_wtime() - t);

    t = omp_get_wtime(); 
    sort(b, 0, N -1, 2);
    printf("Par: %f seconds\n",omp_get_wtime() - t);

    free(a);
    free(b);

    return 0;
}



#include<stdio.h>
#include<omp.h>

int fib(int n,const char*label){
    int tid=omp_get_thread_num();
    printf("thread %d handling %s call:fib(%d)\n",tid,label,n);

    if(n<2) return n;

    int x,y;

    #pragma omp task shared(x)
    x=fib(n-1,"left");

    #pragma omp task shared(y)
    y=fib(n-2,"right");

    #pragma omp taskwait
    return x+y;
}

int main(){
    int n;
    printf("enter number of fibonacci terms:");
    scanf("%d",&n);

    omp_set_nested(1);
    omp_set_num_threads(4);

    double start=omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i=0;i<n;i++){
                printf("\n\n fib(%d)=???\n",i);
                int result=fib(i,"root");
                printf("fib(%d)=%d\n",i,result);
            }
        }
    }
    double end=omp_get_wtime();
    printf("execution time:%f seconds \n",end - start);
    return 0;
}





#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int is_prime(int x) {
    if (x < 2) return 0;
    for (int d = 2; d * d <= x; d++)
        if (x % d == 0) return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <n> <threads>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    omp_set_num_threads(threads);

    int count;
    double t0, t1;

    count = 0;
    t0 = omp_get_wtime();
    printf("Serial primes up to %d:\n", n);
    for (int i = 1; i <= n; i++) {
        if (is_prime(i)) {
            printf("%d ", i);
            count++;
        }
    }
    t1 = omp_get_wtime();
    printf("\nSerial: %d primes, %f sec\n", count, t1 - t0);

    count = 0;
    int *primes = malloc((n + 1) * sizeof(int));
    t0 = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 1; i <= n; i++) {
        if (is_prime(i)) {
            #pragma omp critical
            primes[count++] = i;
        }
    }

    t1 = omp_get_wtime();
    printf("Parallel(%d) primes up to %d:\n", threads, n);
    for (int i = 0; i < count; i++)
        printf("%d ", primes[i]);

    printf("\nParallel: %d primes, %f sec\n", count, t1 - t0);

    free(primes);
    return 0;
}





#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int number = 42;
        MPI_Send(&number, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); 
        printf("Rank 0 sent number %d to Rank 1\n", number);
    } 
    else if (rank == 1) {
        int received;
        MPI_Recv(&received, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
        printf("Rank 1 received number %d from Rank 0\n", received);
    }

    MPI_Finalize();
    return 0;
}





#include <stdio.h>
#include <mpi.h>

int main(){
    int my_rank, comm_sz;
    int data;

    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_sz);

    if(my_rank ==0){
        printf("enter  the number to be broadcasted\n");
        scanf ("%d",&data);
        printf("process 0 broadcasting data=%d\n",data);
    }

    MPI_Bcast(&data,1,MPI_INT,0,MPI_COMM_WORLD);
    printf("process %d received data= %d\n",my_rank,data);

    MPI_Finalize();
    return 0;
}





#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int send_data[8];
    int recv_data;
    int gathered_data[8];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Root process initializing data:\n");
        for (int i = 0; i < size; i++) {
            send_data[i] = i + 1;
            printf("%d ", send_data[i]);
        }
        printf("\n");
    }

    MPI_Scatter(send_data, 1, MPI_INT, &recv_data, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d received %d\n", rank, recv_data);

    recv_data *= 2;

    MPI_Gather(&recv_data, 1, MPI_INT, gathered_data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nData gathered at root after completion:\n");
        for (int i = 0; i < size; i++) {
            printf("%d ", gathered_data[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}





#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    int value;
    int sum, prod, max, min;
    int all_sum, all_prod, all_max, all_min;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    value = rank + 1;
    printf("Process %d has value %d\n", rank, value);

    MPI_Reduce(&value, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&value, &prod, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);
    MPI_Reduce(&value, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&value, &min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n--- Results using MPI_Reduce (available only at root) ---\n");
        printf("Sum = %d\n", sum);
        printf("Product = %d\n", prod);
        printf("Maximum = %d\n", max);
        printf("Minimum = %d\n", min);
    }

    MPI_Allreduce(&value, &all_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &all_prod, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &all_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &all_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    printf("\nProcess %d after MPI_Allreduce:\n", rank);
    printf("  Sum = %d, Product = %d, Max = %d, Min = %d\n",
           all_sum, all_prod, all_max, all_min);

    MPI_Finalize();
    return 0;
}





Program	Compile Command	Run Command
1	gcc program1_static_schedule.c -fopenmp -o p1	./p1
2	gcc program2_mergesort.c -fopenmp -o p2	./p2
3	gcc program3_fibonacci_tasks.c -fopenmp -o p3	./p3
4	gcc program4_primes.c -fopenmp -o p4	./p4
5	mpicc program5_sendrecv.c -o p5	mpirun -np 2 ./p5
7	mpicc program7_bcast.c -o p7	mpirun -np 4 ./p7
8	mpicc program8_scatter_gather.c -o p8	mpirun -np 4 ./p8
9	mpicc program9_reduce_allreduce.c -o p9	mpirun -np 4 ./p9
