Heterosis Network Simulation Package
====================================

This project is a simulation study of Heterosis

It includes a random network model as well as 
hybridization methods and expression simulations


packages
--------
1. graphs, this package includes implementations and graph utility functions to get stats about the graphs.
   - includes:
     + BA-Graph
     + ER-Graph
     + ModGraph - C clusters, P likelihood of edges between clusters, Q edges
       within cluster
     + BA-TF Graph - BA model with extra step for triad formation to promote
       more clustering between nodes.
2. dynamics, includes different heterosis models of dynamics on transcription
   factor networks
   - includes:
     + boolean
     + linear-threshold
     + differential 
3. graphdiff, a set of tools to plot and compare performance of varying graphs
   and dynamics models for measuring performance under hybrid vigor
