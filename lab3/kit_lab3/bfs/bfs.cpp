#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances)
{

#pragma omp parallel for
    for (int i=0; i<frontier->count; i++) {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1)
                           ? g->num_edges
                           : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
            int outgoing = g->outgoing_edges[neighbor];

            if(distances[outgoing]==NOT_VISITED_MARKER)
            if(__sync_bool_compare_and_swap (&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1)==true){
                int index=0;
                index = __sync_fetch_and_add(&new_frontier->count, 1);
                    new_frontier->vertices[index] = outgoing;    
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED  - - can be parallel
    #pragma omp parallel for schedule(static,32) //NOTE if(graph->num_nodes > 1000000)
    for (int i=1; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

//#pragma omp parallel shared(frontier, new_frontier, sol, graph)
{
    while (frontier->count != 0) {


#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        //#pragma omp single
        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        //#pragma omp single
        {
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        }
    }
}
}

void bot_up_step(
    Graph g,
    int* distances, 
    int current_value,
    int &count
    )
{
    int count_ft = 0;
    #pragma omp parallel for schedule(dynamic, g->num_nodes/1000) reduction(+:count_ft)
    for (int node=g->num_nodes-1; node>=0; node--) {
        if(distances[node]==NOT_VISITED_MARKER){
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {

                    if(distances[g->incoming_edges[neighbor]]==current_value){
                        distances[node] = distances[g->incoming_edges[neighbor]] + 1;
                        count_ft++;
                    }
                    
            }
        }
    }
    count = count_ft;
}

void bfs_bottom_up(Graph graph, solution* sol){
    int currentval = 0;
    int count = 1;

    #pragma omp parallel for schedule(static,32)
    for (int i=1; i<graph->num_nodes; i++){
        sol->distances[i] = NOT_VISITED_MARKER;
    }
    // setup frontier with the root node
    sol->distances[ROOT_NODE_ID] = 0;


    while (count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        bot_up_step(graph, sol->distances, currentval, count);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
        currentval++;
    }
}

void bot_up_step_hybrid(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int* distances, 
    int current_value
    )
{
    int count_ft = 0;
    #pragma omp parallel for schedule(dynamic, g->num_nodes/1000)
    for (int node=g->num_nodes-1; node>=0; node--) {
        if(distances[node]==NOT_VISITED_MARKER){
            int start_edge = g->incoming_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->incoming_starts[node + 1];

            // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {

                    if(distances[g->incoming_edges[neighbor]]==current_value){
                        distances[node] = distances[g->incoming_edges[neighbor]] + 1;
                        
                        int index=0;
                        index = __sync_fetch_and_add(&new_frontier->count, 1);
                        new_frontier->vertices[index] = node;
                        break;
                    }
                    
            }
        }
    }
}

void bfs_hybrid(Graph graph, solution* sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;
    int currentval = 0;

    #pragma omp parallel for schedule(static,32)
    for (int i=1; i<graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;


    while (frontier->count != 0) {

#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);
        if(frontier->count<(graph->num_nodes)/4){
            top_down_step(graph, frontier, new_frontier, sol->distances);
        }else{
            bot_up_step_hybrid(graph, frontier, new_frontier, sol->distances, currentval);
            //bot_up_step(graph, sol->distances, currentval, currentval);
        }
        

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers

        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        currentval++;
    }
}
