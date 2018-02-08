#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

#define epsilon 0.00001
#define maxRep 40

__global__ void meanShift(double *x,size_t pitchx,double *y, size_t pitchy,double *ynew,size_t pitchynew,int N,int d,double sigma);
__device__ double calcDist(double *y,size_t pitchy,double *x,size_t pitchx,int d);
__device__ double gausK(double x,double sigma);
double froNorm(double *a,size_t pitcha,double *b,size_t pitchb,int N,int d);
void test(double *y,size_t pitchy,char *testfile,int N,int d);

int main(int argc,char **argv){	

	if(argc!=4){
		printf("Usage: %s (dataset) (test) (sigma) where (dataset) ",argv[0]);
		printf("is the name of the dataset .txt file, (test) is the name of the ");
		printf(".txt test file and (sigma) is the value of sigma for the current dataset\n");
		exit(1);
	}

	struct timeval startwtime, endwtime;
	double time;

	//turn (sigma) input from string to double
	double sigma=atof(argv[3]);

	int i,j; 

	//argv[1] is the (dataset) file
	FILE *file = fopen(argv[1], "r");
	if(file==NULL){
		printf("Couldn't open %s\n",argv[1]);
		exit(1);
	}

	//count the number of points and dimensions of (dataset)
	int d=0,N=0;
	char ch;

	/**dimension and number of points counting found in
	 *https://www.opentechguides.com/how-to/article/c/72/c-file-counts.html
	*/
	while ((ch=getc(file)) != EOF) {
		if ((ch == ' ' || ch == '\n') && N==0) { ++d; }
		
		if (ch == '\n') { ++N; }
	}

	//1 dimension host memory allocation to fit cudaMemcpy2D
	double *y;
	size_t pitchy = sizeof(double) * d;
	y = (double*)malloc(sizeof(double) * N * d);

	double *ynew;
	size_t pitchynew = sizeof(double) * d;
	ynew = (double*)malloc(sizeof(double) * N * d);

	double *x;
	size_t pitchx = sizeof(double) * d;
	x = (double*)malloc(sizeof(double) * N * d);

	double *row_x,*row_y; 
	//return file pointer to the beggining of the file
	fseek(file, 0, SEEK_SET);
	for (i=0;i<N;i++){
		row_x = (double*)((char*)x + i * pitchx );
		row_y = (double*)((char*)y + i * pitchy  );
		for (j=0;j<d;j++){
			fscanf(file,"%lf",&row_x[j]);
			row_y[j]=row_x[j];
		}

	}

	fclose(file);

	//allocate 2d arrays for device memory
	double *d_x;
	double *d_y;
	double *d_ynew;
	size_t d_pitchx,d_pitchy,d_pitchynew;

	cudaMallocPitch((void**)&d_x, &d_pitchx, d * sizeof(double), N);
	cudaMallocPitch((void**)&d_y, &d_pitchy, d * sizeof(double), N);
	cudaMallocPitch((void**)&d_ynew, &d_pitchynew, d * sizeof(double), N);

	//copy data from host to device memory
	cudaMemcpy2D(d_x,d_pitchx,x,pitchx, d * sizeof(double), N, cudaMemcpyHostToDevice);
	cudaMemcpy2D(d_y,d_pitchy,y,pitchy, d * sizeof(double), N, cudaMemcpyHostToDevice);




	int repeats=0;
	double norm;
	double *row_ynew;
	gettimeofday (&startwtime, NULL);

	do{
		meanShift<<<N,d>>>(d_x,d_pitchx,d_y,d_pitchy,d_ynew,d_pitchynew,N,d,sigma);
		cudaMemcpy2D(y, sizeof(double)*d, d_y, d_pitchy, sizeof(double) * d, N, cudaMemcpyDeviceToHost);
		
		//calculate norm of (ynew-y)
		norm = froNorm(y,pitchy,ynew,pitchynew,N,d);
		
		//update ynew after a meanshift iteration
		for (i=0;i<N ;i++){
			row_ynew = (double*)((char*)ynew + i * pitchynew);
			row_y = (double*)((char*)y +i * pitchy);
			for (j=0;j<d;j++){
				row_ynew[j] = row_y[j];
			}
		}
		repeats++;

	}while(norm>epsilon && repeats<maxRep);

	gettimeofday (&endwtime, NULL); 
	
	time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
	+ endwtime.tv_sec - startwtime.tv_sec);
		
	printf("Wall clock time: %f \n", time);

	//argv[2] is the (testfile) name
	test(y,pitchy,argv[2],N,d);

	return 0;

}

__global__ void meanShift(double *x,size_t pitchx,double *y, size_t pitchy,double *ynew,size_t pitchynew,int N,int d,double sigma){
	int index=blockDim.x*blockIdx.x+threadIdx.x;

	if (index<N){
		double sum=0,res=0;
		int j,k;
		
		double* row_y=(double*)((char*)y+index*pitchy);	
		double* row_ynew=(double*)((char*)ynew+index*pitchynew);
		
		//initialize ynew
		for(k=0;k<d;k++)
			row_ynew[k]=0;
				
		for(j=0;j<N;j++){
			double* row_x=(double*)((char*)x+j*pitchx);
			
			if(calcDist(row_y,pitchy,row_x,pitchx,d)<sigma*sigma){
				double temp=0;
				for(k=0;k<d;k++){
					temp=(row_y[k]-row_x[k])*(row_y[k]-row_x[k])+temp;
					//temp is the square of norm2(y_i-x_j)
				}	
				res=gausK(temp,sigma);
				
				for(k=0;k<d;k++){
					row_ynew[k]=row_ynew[k]+row_x[k]*res;
				}
				sum=sum+res;
				//calculating denominator of ynew_i
			}	
		}
		for(k=0;k<d;k++){
				row_ynew[k]=row_ynew[k]/sum;
		}
		//update y from all threads
		for(k=0;k<d;k++){
				row_y[k]=row_ynew[k];
		}
	}
}

//calculate distance between x and y
__device__ double calcDist(double *y,size_t pitchy,double *x,size_t pitchx,int d){
	double sum = 0;
	int l;

	for (l=0;l<d;l++){
		sum = sum + (y[l]-x[l])*(y[l]-x[l]);
	}

	return sqrt(sum);
}

__device__ double gausK(double x,double sigma){
	double f;
	f = exp(-x/(2*(sigma*sigma)));

	return f;
}

//calculate frobenius norm of (a-b)
double froNorm(double *a,size_t pitcha,double *b,size_t pitchb,int N,int d){
	int i,j;
	double sum=0;
	double *row_b,*row_a;

	for (i=0;i<N;i++){
		row_a = (double*)((char*)a + i * pitcha);
		row_b = (double*)((char*)b + i * pitchb);
		for (j=0;j<d;j++){
			sum = sum + (row_a[j]-row_b[j])*(row_a[j]-row_b[j]);
		}
	}

	return sqrt(sum);
}

void test(double *y,size_t pitchy,char *testfile,int N,int d){
	int i,j;
	double **test;

	//memory allocation for test input
	test =(double **) malloc(sizeof(double*)*N);

	for (i=0;i<N;i++){
		test[i] = (double *)malloc(sizeof(double)*d);
	}
	FILE *file = fopen(testfile, "r"); 
	if(file==NULL){
		printf("Couldn't open %s\n",testfile);
		exit(1);
	}
	
	for (i=0;i<N;i++){
		for (j=0;j<d;j++){
			fscanf(file,"%lf",&test[i][j]);
		}
	}

	//compare the arrays
	int failed=0;
	for (i=0;i<N;i++){
		double* row_y=(double*)((char*)y+i*pitchy);	
		for (j=0;j<d;j++){
			//check if relative error to matlab output is small
			if (fabs(row_y[j]-(double)test[i][j])/fabs((double)test[i][j]) > 0.1)
				failed++;
		}
	}

	//check if a small percentage of the result is wrong
	if((double)(d*N-failed)/(double)(d*N)*100<95.0)
		printf("Test failed!\n");
	else
		printf("Test passed!\n");

	fclose(file);
}

