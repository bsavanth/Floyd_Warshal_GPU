#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <cuda.h>
#include <ctime>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define pb push_back 
#define all(c) (c).begin(),(c).end()
#pragma comment(lib, "winmm.lib")
#define _CRTDBG_MAP_ALLOC
using namespace std;

#define _DTH cudaMemcpyDeviceToHost
#define _HTD cudaMemcpyHostToDevice

//these can be altered on user depending on data set and type of operation(random test, read from file etc)
#define BLOCK_SIZE 256
#define RANGE 997
#define RANDOM_GSIZE 132
#define FILE_GSIZE 8298//the number of edges in Wiki-Vote.txt if the file test is run
#define INF 9999
#define DO_TEST_RANDOM 1
#define con 1024.0
#define DO_TEST_FROM_FILE 0

//typedef for vector used in path reconstruction
typedef pair<pair<int,int>,int> Piii;

//forward function declarations
void _CPU_Floyd(int *G,int *Gpath,int N);
void _showPath(int start,int end,const vector<Piii> &path,const int *D,const int N);
bool _getPath(int curEdge, int nxtEdge,vector<Piii> &path,const int *D, const int *Dpath,const int N);
void _get_full_paths(const int *D, const int *Dpath, const int N);
void verify_results(int*, int*, int, int);
//CUDA GPU kernel/functions forward declaration
__global__ void _Wake_GPU(int reps);
__global__ void _GPU_Floyd_kernel(int k, int *G,int *P, int N);
void _GPU_Floyd(int *H_G, int *H_Gpath, const int N);
void used_gpu_mem();

//other optional utility functions
int _read_from_file(int *G,const int N);
void _generateRandomGraph(int *G, int N, int range, int density);
void _generate_result_file(bool success, double cpu_time, double gpu_time, int N);

 std::map<int,int> airlist;
 int idmat[132];
 int id[132];


int main(){
	
	srand(time(NULL));

	if(DO_TEST_RANDOM){//will use the #define(s) to init a random adjacency Matrix of RANDOM_GSIZE size
		const int NumBytes=RANDOM_GSIZE*RANDOM_GSIZE*sizeof(int);
		//host allocations to create Adjancency matrix and result matrices with path matrices
		int *OrigGraph=(int *)malloc(NumBytes);//will be original Adjancency matrix, will NOT be changed
		int *H_G=(int *)malloc(NumBytes);
		int *H_Gpath=(int *)malloc(NumBytes);
		int *D_G=(int *)malloc(NumBytes);
		int *D_Gpath=(int *)malloc(NumBytes);

		_generateRandomGraph(OrigGraph,RANDOM_GSIZE,RANGE,25);//init graph with values
                cout<<"\n------------------------------------------initialization---------------------------------"<<endl;
		cout<<"Successfully created random highly connected graph in adjacency Matrix form with "<<RANDOM_GSIZE*RANDOM_GSIZE<< " elements.\n";
		cout<<"Also created 2 pairs of distinct result Matrices to store the respective results of the CPU results and the GPU results.\n";
		for(int i=0;i<RANDOM_GSIZE*RANDOM_GSIZE;i++){//copy for use in computation
			H_G[i]=D_G[i]=OrigGraph[i];//copy for use in computation
			H_Gpath[i]=D_Gpath[i]=-1;//set to all negative ones for use in path construction
		}
		double cpu_time=0,gpu_time=0;
                cout<<"\n-----------------------------------------algorithm on cpu-------------------------------"<<endl;
		cout<<"\nFloyd-Warshall on CPU underway:\n";

		double t1,t2;
		t1=clock(); 
		_CPU_Floyd(H_G,H_Gpath,RANDOM_GSIZE);//find shortest paths (with path construction) on serial CPU (Intel i7 3770 3.9 ghz)
		t2=clock();
		cpu_time = ((double)(t2 - t1))/CLOCKS_PER_SEC;

		printf("CPU Timing %fs\n\n", cpu_time);
	
		//wake up GPU from idle
                cout<<"\n-----------------------------------------algorithm on Gpu-------------------------------"<<endl;
		cout<<"\nFloyd-Warshall on GPU underway:\n";
		_Wake_GPU<<<1,BLOCK_SIZE>>>(32);

		//call host function which will copy all info to device and run CUDA kernels

		t1 =clock();
		_GPU_Floyd(D_G,D_Gpath,RANDOM_GSIZE);
		t2 = clock();
		gpu_time = ((double)(t2 - t1))/CLOCKS_PER_SEC;
		printf("GPU Timing(including all device-host, host-device copies, device allocations and freeing of device memory): %fs\n\n", gpu_time);
                cout<"\n\n";
                cout<<"****************"<<endl;
                cout<< "GPU is "<<(cpu_time/gpu_time)<<" times faster than CPU"<<"\n\n";
                cout<<"****************"<<endl;
                cout<<"\n\n-----------------------------------------utilities running-------------------------------"<<endl;
                verify_results(H_G, D_G, RANDOM_GSIZE,NumBytes);
                used_gpu_mem();
                 
             /*   for(int i=0;i<132;i++)
{
	cout<<id[i]<<"\t"<<i<<endl ;
} */
		//check shortest path from one vertex to another

		/******GPU Graph********/
		_get_full_paths(D_G,D_Gpath,RANDOM_GSIZE);//find out exact step-by-step shortest paths between vertices(if such a path exists)

		/******CPU Graph*******/
	     _get_full_paths(H_G,H_Gpath,RANDOM_GSIZE);////find out exact step-by-step shortest paths between vertices(if such a path exists) 	

		//_generate_result_file( bool(same_adj_Matrix==0 && same_path_Matrix==0),cpu_time,gpu_time,RANDOM_GSIZE);

	/////////////////////////////////////////////////////////////////////////////////


		free(OrigGraph);
		free(H_G);
		free(H_Gpath);
		free(D_G);
		free(D_Gpath);
	}

	return 0;
}


void _CPU_Floyd(int *G,int *Gpath,int N){//standard N^3 algo
	for(int k=0;k<N;++k)for(int i=0;i<N;++i)for(int j=0;j<N;++j){
		int curloc=i*N+j,loca=i*N+k,locb=k*N+j;
		if(G[curloc]>(G[loca]+G[locb])){
			G[curloc]=(G[loca]+G[locb]);
			Gpath[curloc]=k;
		}
	}
	
       cout<"\n\n\n";
       
}

void _showPath(int start,int end,const vector<Piii> &path,const int *D,const int N){
       /* for(int i=0;i<132;i++)
       {
             
           cout<<i<<"\t"<<id[i]<<endl;
       }
         */                      


	cout<<"\nHere is the shortest cost path from "<<id[start]<< " to "<<id[end]<<", at a total cost of "<<D[start*N+end]<<".\n";
	for(int i=path.size()-1;i>=0;--i){
		cout<<id[path[i].first.first]<<"-->"<<id[path[i].first.second]<<'\n';
	}
	cout<<'\n';
}

bool _getPath(int curEdge, int nxtEdge,vector<Piii> &path,const int *D, const int *Dpath,const int N){
	int curIdx=curEdge*N+nxtEdge;
	if(D[curIdx]>=INF)return false;
	if(Dpath[curIdx]==-1){//end of backwards retracement
		path.push_back(make_pair(make_pair(curEdge,nxtEdge),D[curIdx]));
		return true;
	}else{//record last edge cost and move backwards
		path.push_back(make_pair(make_pair(Dpath[curIdx],nxtEdge),D[Dpath[curIdx]*N+nxtEdge]));
		return _getPath(curEdge,Dpath[curIdx],path,D,Dpath,N);
	}
}

void _get_full_paths(const int *D, const int *Dpath, const int N){
	int start_vertex=-1,end_vertex=-1;
	vector<Piii> path;
        cout<<"\n-----------------------------------------path-tracking-------------------------------"<<endl;
	do{
		path.clear();
		cout<<"Enter start vertex #:";
		cin>>start_vertex;
                 map<int,int>::iterator t1 = airlist.find(start_vertex);
                start_vertex= t1->second;
		cout<<"Enter dest vertex(enter negative number to exit) #:";
		cin>>end_vertex;
                 map<int,int>::iterator t2 = airlist.find(end_vertex);
                end_vertex= t2->second;
		
                if(start_vertex<0 || start_vertex>=N || end_vertex<0 || end_vertex>=N)return;

		if(_getPath(start_vertex, end_vertex,path,D,Dpath,N)){
			_showPath(start_vertex,end_vertex,path,D,N);

		}else{
			cout<<"\nThere does not exist valid a path between "<<id[start_vertex]<<" , and "<<id[end_vertex]<<'\n';

		}
	}while(0);
}

__global__ void _Wake_GPU(int reps){
	int idx=blockIdx.x*blockDim.x + threadIdx.x;
	if(idx>=reps)return;
}

__global__ void _GPU_Floyd_kernel(int k, int *G,int *P, int N){//G will be the adjacency matrix, P will be path matrix
	int col=blockIdx.x*blockDim.x + threadIdx.x;
	if(col>=N)return;
	int idx=N*blockIdx.y+col;

	__shared__ int best;
	if(threadIdx.x==0)
		best=G[N*blockIdx.y+k];
	__syncthreads();
	if(best==INF)return;
	int tmp_b=G[k*N+col];
	if(tmp_b==INF)return;
	int cur=best+tmp_b;
	if(cur<G[idx]){
		G[idx]=cur;
		P[idx]=k;
	}
        __syncthreads();

}
void _GPU_Floyd(int *H_G, int *H_Gpath, const int N){
	//allocate device memory and copy graph data from host
	int *dG,*dP;
	int numBytes=N*N*sizeof(int);
	cudaError_t err=cudaMalloc((int **)&dG,numBytes);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((int **)&dP,numBytes);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	//copy from host to device graph info
	err=cudaMemcpy(dG,H_G,numBytes,_HTD);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(dP,H_Gpath,numBytes,_HTD);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	dim3 dimGrid((N+BLOCK_SIZE-1)/BLOCK_SIZE,N);

	for(int k=0;k<N;k++){//main loop

		_GPU_Floyd_kernel<<<dimGrid,BLOCK_SIZE>>>(k,dG,dP,N);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	}
	//copy back memory
	err=cudaMemcpy(H_G,dG,numBytes,_DTH);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMemcpy(H_Gpath,dP,numBytes,_DTH);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
         

          

             
/*free device memory
	err=cudaFree(dG);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(dP);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
*/ }






//--------------method to verify results of gpu and cpu---------------------------//

  void verify_results( int *G, int *D, int N, int k)
 {


       cout<<"\nVerifying results of final adjacency Matrix and Path Matrix.\n";

		int same_adj_Matrix = memcmp(G,D,k);
		if(same_adj_Matrix==0){
			cout<<"Adjacency Matrices Equal!\n";
		}else
			cout<<"Adjacency Matrices Not Equal!\n";


       cout<<"\n\n";

  }

 

//---------------------------------------------------------------------------------//




void used_gpu_mem()
 {
  
  cout<<"\n-----------------------------------------Memory considerations-------------------------------"<<endl;      
   float free_m, total_m, used_m;
   size_t free_t, total_t;
   cudaMemGetInfo(&free_t, &total_t);
   free_m = (uint)free_t/(con*con);
   total_m= (uint) total_t/(con*con);
   used_m=total_m-free_m;
  cout<<"\n\nMEMORY USED BY GPU FOR THE ABOVE COMPUTATION IS:"<<used_m<<"\n\n";
   cout<<"FREE MEMORY IN GPU IS:"<<free_m<<"\n\n";
 





 }

/////////////////////////// GRAPH GENERATION /////////////////////////////////




void _generateRandomGraph(int *G,int N,int range, int density){//density will be between 0 and 100, indication the % of number of directed edges in graph
	//range will be the range of edge weighting of directed edges

	
  

 int count=0;

for(int i=0;i<N;i++)
 {
    
     for(int j=0;j<N;j++)
      {
        count++;
	if(i==j)
         {
            G[i*N+j]=0;
            continue;
	}
        else
         {
            G[i*N+j]=INF;
         }
         
      }
     
 }
 
 //cout<<"\n\n\n"<<count<<"\n\n\n";

 //map and file stream initialization

  ifstream in;
  in.open("input.txt");
  string line;

 
  ifstream list;
 list.open("airportsList.txt");
 string lineiter;
 
 int i=0;
  while(getline(list, lineiter))
{  
  
   string name;
   istringstream itering(lineiter);
   itering >>name; 
   itering>>ws;
   itering>>id[i];
   airlist.insert(std::make_pair(id[i], i));
   // cout<<name<<"\t"<<id[i]<<endl;
    i++;
}
int j=0; 
while(j<132)
{ 
 int current=idmat[j];
 map<int,int>::iterator k=airlist.find(current);
cout<<k->first<<"\t"<<k->second;
//cout<<current; 
cout<<endl;
 j++;
}
while(getline(in, line))
 {
  

  int srcdest; int weight,route; 
  replace(line.begin(),line.end(),',',' '); 
  int source, destination=0, len;
  istringstream iter(line);
  iter>>srcdest>>weight>>route;
  len=floor(log10(abs(srcdest))) + 1;
  source = srcdest / pow(10, len / 2);
  destination = srcdest - source*pow(10, len / 2);
//  cout<<(source-10000)<<" "<<(destination-10000)<<" "<<weight<<" "<<route<<endl;
  map<int,int>::iterator h = airlist.find(source);
  map<int,int>::iterator l=airlist.find(destination);
  int dupsource=h->second;
  int dupdest= l->second;
//  cout<<dupsource<<"\t"<<dupdest<<"\t"<<weight<<endl;
 if(route==1)
  G[dupsource*N+dupdest]= weight;
}
  
int count1;

for(i=0;i<N*N;i++)
{
 if(G[i]>0 && G[i]<INF)
  {
    count1++;
  }
} 

cout<<"\nNODES IN THE GRAPH #: "<<N<<endl;
cout<<"\nEDGES IN THE GRAPH #: "<<count<<endl;



}

int _read_from_file(int *G,const int N){//reads in edge list from file
	int num_edges=0;
	
	ifstream readfile;//enable stream for reading file
	readfile.open("Wiki-Vote.txt");
	assert(readfile.good());//make sure it finds the file & file is
	string line;
	int v0,v1;
	while(getline(readfile,line)){
		istringstream linestream(line);
		linestream>>v0>>v1;
		G[v0*N+v1]=1;
		num_edges++;
	}
	readfile.close();
	return num_edges;
}

void _generate_result_file(bool success, double cpu_time,double gpu_time, int N){

	if(!success){
		cout<<"Error in calculation!\n";
		return;
	}else{
		ofstream myfile;
		myfile.open("Floyd-Warshall_result.txt");
		myfile<<"Success! The GPU Floyd-Warshall result and the CPU Floyd-Warshall results are identical(both final adjacency matrix and path matrix).\n\n";
		myfile<<"N= "<<N<<" , and the total number of elements(for Adjacency Matrix and Path Matrix) was "<<N*N<<" .\n";
		myfile<<"Matrices are int full dense format(row major) with a minimum of "<<(N*N)/4<<" valid directed edges.\n\n";
		myfile<<"The CPU timing for all was "<<float(cpu_time)<<" seconds, and the GPU timing(including all device memory operations(allocations,copies etc) ) for all was "<<float(gpu_time)<<" seconds.\n";
		myfile<<"The GPU result was "<<float(cpu_time)/float(gpu_time)<<" faster than the CPU version.\n";
		myfile.close();
	}
}
