#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


/*
  TODO STUDENTS: Implement the page rank algorithm here.  You
  are expected to parallelize the algorithm using openMP.  Your
  solution may need to allocate (and free) temporary arrays.

  Basic page rank pseudocode is provided below to get you started:

  // initialization: see example code above
  score_old[vi] = 1/numNodes;

  while (!converged) {

    // compute score_new[vi] for all nodes vi:
    score_new[vi] = sum over all nodes vj reachable from incoming edges
                      { score_old[vj] / number of edges leaving vj  }
    score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

    score_new[vi] += sum over all nodes v in graph with no outgoing edges
                      { damping * score_old[v] / numNodes }

    // compute how much per-node scores have changed
    // quit once algorithm has converged

    global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
    converged = (global_diff < convergence)
  }

*/
// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{
  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs
  double global_diff=0.0;
  //double score_old[g->num_nodes],score_new[g->num_nodes];
  double *score_new;
  double buffer_score;
  Vertex* no_outgoing_nodes;
  int no_counter=0,index;
  score_new = (double*)malloc(sizeof(double) * g->num_nodes);
  no_outgoing_nodes = (Vertex*)malloc(sizeof(Vertex) * g->num_nodes);
  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;

  #pragma omp parallel for nowait
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
    score_new[i] = 0.0;
    if(outgoing_size(g,i)==0){
      index = __sync_fetch_and_add(&no_counter, 1);
      no_outgoing_nodes[index]=i;
    }
  }


    // initialization: see example code above
    int first=1;

    // quit once algorithm has converged
    #pragma omp parallel
    {
      do{
        
        #pragma omp single nowait
        buffer_score=0.0;
        #pragma omp single nowait
        global_diff=0.0;

        if(first==0){
          #pragma omp for       
          for(int i=0;i<numNodes;i++){
            solution[i] = score_new[i];
            score_new[i] = 0.0;
          }//FIXME barrier
        }

        if(no_counter>0){
          #pragma omp for reduction(+:buffer_score)
          for(int k=0;k<no_counter;k++){
            buffer_score+=solution[no_outgoing_nodes[k]];
          }
          #pragma omp single nowait
          buffer_score=damping*buffer_score/numNodes;
        }
        //NOTE pragma parallel for reduction de soma no score new
        //print_graph(g);
        // compute score_new[vi] for all nodes vi:
        /*SECTION score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }*/
        /*SECTION score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }*/
        #pragma omp for nowait
        for(int i=0;i<numNodes;i++){
          const Vertex* pointer=incoming_begin(g,i);
          for(int j=0;j<incoming_size(g,i);j++){
            score_new[i] += solution[pointer[j]]/ outgoing_size(g,pointer[j]);
            //printf("[%d]: begin->%d\t%f/%d=%f\n",i,pointer[j],score_old[pointer[j]],outgoing_size(g,pointer[j]),score_new[i]);
          }
          score_new[i] = ((damping * score_new[i]) + (1.0-damping) / numNodes) + buffer_score; //NOTE end parallel section
        }

        //SECTION global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
        // compute how much per-node scores have changed
        #pragma omp for reduction(+:global_diff)
        for(int i=0;i<numNodes;i++){
          if(score_new[i] - solution[i]<0.0){
            global_diff -= score_new[i] - solution[i];
          }else{
            global_diff += score_new[i] - solution[i];
          }
        }
          
        #pragma omp single
        first=0;
      }while (global_diff >= convergence);
    }

    //SECTION set solution
    #pragma omp parallel for
    for (int i = 0; i < numNodes; i++) {
      solution[i] = score_new[i];
    }
    free(score_new);
    free(no_outgoing_nodes);
}
