#!/usr/bin/env python
# coding: utf-8

# # Link Prediction

# Caricamento della rete completa di Napoli da file

# In[1]:


import pickle
import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import os
import linkpred
from random import randint
from copy import deepcopy

current_dir = os.getcwd()
picture_dir = os.path.join(current_dir, 'grafici')
objects_dir = os.path.join(current_dir, 'objects')
network_file_name = os.path.join(objects_dir, 'napoli.network')

with open(network_file_name, 'rb') as napoli_network_file:
    g = pickle.load(napoli_network_file)

#Erdos-Renyi random Graph
density = g.number_of_edges()/(g.number_of_nodes()*(g.number_of_nodes()-1))
er = nx.erdos_renyi_graph(g.number_of_nodes(), density)


# Si definisce un grafo diretto semplice in modo da non avere problemi derivanti dall'utilizzo del multigraph nella libreria linkpred. Si inseriscon i nodi e i link dopodichè si salva e si ricarica da file edgelist

# In[2]:


G = nx.DiGraph()
G.add_nodes_from(g.nodes())
G.add_edges_from(g.edges())
G.remove_edges_from(G.selfloop_edges())


# ## Katz

# Si rimuovono n_test link dal network, si fa link prediction e si confrontano i primi n_test link predetti dall'algoritmo con gli n_test link rimossi. Si generano poi n_test link in maniera random e si confrontano con gli n_test link rimossi. Si confrontano infine i rusultati in modo da verificare la predizione random

# In[4]:


arr_edges = np.array(G.edges())
n_edges = G.number_of_edges()
n_nodes = G.number_of_nodes()
n_run = 50
n_test = 1000
avg_accuracy = 0
avg_accuracy_random = 0

for i in range(n_run):
    G_reduc = deepcopy(G)
    
    #estrazione link da testare e successiva rimozione dal network
    edge_test = [tuple(arr_edges[randint(0,n_nodes-1)]) for i in range(n_test)]
    G_reduc.remove_edges_from(edge_test)
    
    #applicazione algoritmo di link prediction
    katz = linkpred.predictors.Katz(G_reduc, excluded=G_reduc.edges())
    katz_results = katz.predict()
    
    #estrazione primi link predetti nel modello con la rete originale
    top = katz_results.top(n_test)
    predicted_edges = [tuple(link) for link in list(top.keys())]
    
    
    #calcoloc frazione link predetti correttamente tramite il algoritmo
    accuracy = len(set(predicted_edges).intersection(set(edge_test)))/n_test
    avg_accuracy += accuracy
    
    #estrazione link random non appartenenti al network ridotto
    random_edges = list()
    n_nodes = G_reduc.number_of_nodes()
    arr_nodes = np.array(G_reduc.nodes())
    k=0
    while k != n_test:
        edge = (arr_nodes[randint(0,n_nodes-1)], arr_nodes[randint(0,n_nodes-1)])
        if edge not in list(G_reduc.edges()):
            random_edges.append(edge)
            k=k+1
    
    #calcoloc frazione link predetti correttamente tramite estrazione random
    accuracy_random = len(set(random_edges).intersection(set(edge_test)))/n_test
    avg_accuracy_random += accuracy_random
    

avg_accuracy = avg_accuracy/n_run
avg_accuracy_random = avg_accuracy_random/n_run

print('Average percentage of right predictions with link prediction algorithm: {}%'.format(avg_accuracy*100))
print('Average percentage of right predictions with random edge generation: {}%'.format(avg_accuracy_random*100))


# ## Common Neighbours

# Si esegue l'algoritmo Common Neighbours sul grafo G

# In[5]:


arr_edges = np.array(G.edges())
n_edges = G.number_of_edges()
n_nodes = G.number_of_nodes()
n_run = 50
n_test = 1000
avg_accuracy = 0
avg_accuracy_random = 0

for i in range(n_run):
    G_reduc = deepcopy(G)
    
    #estrazione link da testare e successiva rimozione dal network
    edge_test = [tuple(arr_edges[randint(0,n_nodes-1)]) for i in range(n_test)]
    G_reduc.remove_edges_from(edge_test)
    
    #applicazione algoritmo di link prediction
    commonneig = linkpred.predictors.CommonNeighbours(G, excluded=G.edges())
    common_results = commonneig.predict()
    
    #estrazione primi link predetti nel modello con la rete originale
    top = common_results.top(n_test)
    predicted_edges = [tuple(link) for link in list(top.keys())]
    
    
    #calcoloc frazione link predetti correttamente tramite l'algoritmo
    accuracy = len(set(predicted_edges).intersection(set(edge_test)))/n_test
    avg_accuracy += accuracy
    
    #estrazione link random non appartenenti al network ridotto
    random_edges = list()
    n_nodes = G_reduc.number_of_nodes()
    arr_nodes = np.array(G_reduc.nodes())
    k=0
    while k != n_test:
        edge = (arr_nodes[randint(0,n_nodes-1)], arr_nodes[randint(0,n_nodes-1)])
        if edge not in list(G_reduc.edges()):
            random_edges.append(edge)
            k=k+1
    
    #calcoloc frazione link predetti correttamente tramite estrazione random
    accuracy_random = len(set(random_edges).intersection(set(edge_test)))/n_test
    avg_accuracy_random += accuracy_random
    

avg_accuracy = avg_accuracy/n_run
avg_accuracy_random = avg_accuracy_random/n_run

print('Average percentage of right predictions with link prediction algorithm: {}%'.format(avg_accuracy*100))
print('Average percentage of right predictions with random edge generation: {}%'.format(avg_accuracy_random*100))


# ## Jaccard

# Si esegue l'algoritmo Common Neighbours sul grafo G

# In[6]:


arr_edges = np.array(G.edges())
n_edges = G.number_of_edges()
n_nodes = G.number_of_nodes()
n_run = 50
n_test = 1000
avg_accuracy = 0
avg_accuracy_random = 0

for i in range(n_run):
    G_reduc = deepcopy(G)
    
    #estrazione link da testare e successiva rimozione dal network
    edge_test = [tuple(arr_edges[randint(0,n_nodes-1)]) for i in range(n_test)]
    G_reduc.remove_edges_from(edge_test)
    
    #applicazione algoritmo di link prediction
    jaccard = linkpred.predictors.Jaccard(G, excluded=G.edges())
    jaccard_results = jaccard.predict()
    
    #estrazione primi link predetti nel modello con la rete originale
    top = jaccard_results.top(n_test)
    predicted_edges = [tuple(link) for link in list(top.keys())]
    
    
    #calcoloc frazione link predetti correttamente tramite il algoritmo
    accuracy = len(set(predicted_edges).intersection(set(edge_test)))/n_test
    avg_accuracy += accuracy
    
    #estrazione link random non appartenenti al network ridotto
    random_edges = list()
    n_nodes = G_reduc.number_of_nodes()
    arr_nodes = np.array(G_reduc.nodes())
    k=0
    while k != n_test:
        edge = (arr_nodes[randint(0,n_nodes-1)], arr_nodes[randint(0,n_nodes-1)])
        if edge not in list(G_reduc.edges()):
            random_edges.append(edge)
            k=k+1
    
    #calcoloc frazione link predetti correttamente tramite estrazione random
    accuracy_random = len(set(random_edges).intersection(set(edge_test)))/n_test
    avg_accuracy_random += accuracy_random
    

avg_accuracy = avg_accuracy/n_run
avg_accuracy_random = avg_accuracy_random/n_run

print('Average percentage of right predictions with link prediction algorithm: {}%'.format(avg_accuracy*100))
print('Average percentage of right predictions with random edge generation: {}%'.format(avg_accuracy_random*100))

