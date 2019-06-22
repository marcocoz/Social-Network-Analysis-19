#!/usr/bin/env python
# coding: utf-8

# # Community Discovery

# In[1]:


import pickle
import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import folium
from cdlib import algorithms
import os
import geopandas as gpd
from descartes.patch import PolygonPatch
from random import random
from copy import deepcopy

current_dir = os.getcwd()
picture_dir = os.path.join(current_dir, 'grafici')
objects_dir = os.path.join(current_dir, 'objects')
network_file_name = os.path.join(objects_dir, 'napoli.network')

with open(network_file_name, 'rb') as napoli_network_file:
    g = pickle.load(napoli_network_file)
    
nodes, edges = ox.graph_to_gdfs(g)

"""rinomino i nodi con degli interi più piccoli in modo che siano convertibili in degli interi di C
questo passaggio è necessario per far andare certi algoritmi che altrimenti presentano il seguente errore
OverflowError: integer too large for conversion to C int"""

mapping = dict()
for i in range(len(nodes)):
    mapping[nodes.index[i]] = i

g1 = nx.relabel_nodes(g, mapping)
nodes1, edges1 = ox.graph_to_gdfs(g1)


# Si definisce una nuova misura di accuracy. Per ogni quartiere viene individuato la community maggiore (ovvero quella con più nodi); in particolare a noi interessa il numero di nodi che appartegono alla community maggiore. Si sommano tutti questi valori. Successivamente si ripete lo stesso procedimento ma partendo dalla community e andando ad individuare il quartiere maggiore. A questo punto si sommano i due valori ottenuti e si divide tutto per il numero di nodi. La misura ottenuta può assumere valori tra 0 e 1; in particolare assume il valore 1 se e solo se le community corrispondono esattamente ai quartieri. 

# In[2]:


def accuracy_comunity(community_column):
    df = nodes1.groupby('quartiere')
    series = df[community_column].value_counts()
    right_quartier = series.groupby(level=0).max().sum()
    right_community = series.groupby(level=1).max().sum()
    accuracy = (right_quartier + right_community) / (2 * len(nodes1))
    return accuracy


# Si applicano gli algoritmi di community discovery e si salvano i risultati su file.
# 
# ATTENZIONE: richiede qualche minuto
# 
# CONSIGLIO: passare alla cella successiva che carica i risultati da file

# In[30]:


accuracy_spinglass = 0
accuracy_eigenvector = 0
accuracy_leiden = 0
accuracy_cpm = 0
accuracy_rber_pots = 0

for i in range(10):
    result_spinglass_tmp = algorithms.spinglass(g1)
    result_eigenvector_tmp = algorithms.eigenvector(g1)
    result_leiden_tmp = algorithms.leiden(g1)
    result_cpm_tmp = algorithms.cpm(g1, resolution_parameter=.00018)
    result_rber_pots_tmp = algorithms.rber_pots(g1, resolution_parameter=.32)
    
    #definizione colonne che servono per calcolare l'accuracy
    nodes1['community_spinglass'] = -1
    for i in range(len(result_spinglass_tmp.communities)):
        for j in result_spinglass_tmp.communities[i]:
            nodes1.loc[j,'community_spinglass'] = i
    nodes1['community_eigenvector'] = -1
    for i in range(len(result_eigenvector_tmp.communities)):
        for j in result_eigenvector_tmp.communities[i]:
            nodes1.loc[j,'community_eigenvector'] = i
    nodes1['community_leiden'] = -1
    for i in range(len(result_leiden_tmp.communities)):
        for j in result_leiden_tmp.communities[i]:
            nodes1.loc[j,'community_leiden'] = i
    nodes1['community_cpm'] = -1
    for i in range(len(result_cpm_tmp.communities)):
        for j in result_cpm_tmp.communities[i]:
            nodes1.loc[j,'community_cpm'] = i
    nodes1['community_rber_pots'] = -1
    for i in range(len(result_rber_pots_tmp.communities)):
        for j in result_rber_pots_tmp.communities[i]:
            nodes1.loc[j,'community_rber_pots'] = i
    
    #calcolo accuracy per ogni algoritmo
    accuracy_spinglass_tmp = accuracy_comunity('community_spinglass')
    accuracy_eigenvector_tmp = accuracy_comunity('community_eigenvector')
    accuracy_leiden_tmp = accuracy_comunity('community_leiden')
    accuracy_cpm_tmp = accuracy_comunity('community_cpm')
    accuracy_rber_pots_tmp = accuracy_comunity('community_rber_pots')
    
    #confronto con risultati precedenti
    if accuracy_spinglass_tmp > accuracy_spinglass:
        accuracy_spinglass = accuracy_spinglass_tmp
        result_spinglass = deepcopy(result_spinglass_tmp)    
    if accuracy_eigenvector_tmp > accuracy_eigenvector:
        accuracy_eigenvector = accuracy_eigenvector_tmp
        result_eigenvector = deepcopy(result_eigenvector_tmp)
    if accuracy_leiden_tmp > accuracy_leiden:
        accuracy_leiden = accuracy_leiden_tmp
        result_leiden = deepcopy(result_leiden_tmp)
    if accuracy_cpm_tmp > accuracy_cpm:
        accuracy_cpm = accuracy_cpm_tmp
        result_cpm = deepcopy(result_cpm_tmp)
    if accuracy_rber_pots_tmp > accuracy_rber_pots:
        accuracy_rber_pots = accuracy_rber_pots_tmp
        result_rber_pots = deepcopy(result_rber_pots_tmp)
        
#salvataggio risultati su file
result_file_name = os.path.join(objects_dir, 'result_spinglass')
with open(result_file_name, 'wb') as result_file:
    pickle.dump(result_spinglass, result_file)
result_file_name = os.path.join(objects_dir, 'result_eigenvector')
with open(result_file_name, 'wb') as result_file:
    pickle.dump(result_eigenvector, result_file)
result_file_name = os.path.join(objects_dir, 'result_leiden')
with open(result_file_name, 'wb') as result_file:
    pickle.dump(result_leiden, result_file)
result_file_name = os.path.join(objects_dir, 'result_cpm')
with open(result_file_name, 'wb') as result_file:
    pickle.dump(result_cpm, result_file)
result_file_name = os.path.join(objects_dir, 'result_rber_pots')
with open(result_file_name, 'wb') as result_file:
    pickle.dump(result_rber_pots, result_file)


# Caricamento dei risultati da file

# In[4]:


result_file_name = os.path.join(objects_dir, 'result_spinglass')
with open(result_file_name, 'rb') as result_file:
    result_spinglass = pickle.load(result_file)
result_file_name = os.path.join(objects_dir, 'result_eigenvector')
with open(result_file_name, 'rb') as result_file:
    result_eigenvector = pickle.load(result_file)
result_file_name = os.path.join(objects_dir, 'result_leiden')
with open(result_file_name, 'rb') as result_file:
    result_leiden = pickle.load(result_file)
result_file_name = os.path.join(objects_dir, 'result_cpm')
with open(result_file_name, 'rb') as result_file:
    result_cpm = pickle.load(result_file)
result_file_name = os.path.join(objects_dir, 'result_rber_pots')
with open(result_file_name, 'rb') as result_file:
    result_rber_pots = pickle.load(result_file)


# Si attribuisce ad ogni nodo la community corrispondente. I parametri sono stati stimati manualmente in base al numero di community che trovava l'algoritmo.

# In[32]:


nodes1['community_spinglass'] = -1

for i in range(len(result_spinglass.communities)):
    for j in result_spinglass.communities[i]:
        nodes1.loc[j,'community_spinglass'] = i
        
nodes1['community_eigenvector'] = -1

for i in range(len(result_eigenvector.communities)):
    for j in result_eigenvector.communities[i]:
        nodes1.loc[j,'community_eigenvector'] = i
        
nodes1['community_leiden'] = -1

for i in range(len(result_leiden.communities)):
    for j in result_leiden.communities[i]:
        nodes1.loc[j,'community_leiden'] = i
        
nodes1['community_cpm'] = -1

for i in range(len(result_cpm.communities)):
    for j in result_cpm.communities[i]:
        nodes1.loc[j,'community_cpm'] = i
        
nodes1['community_rber_pots'] = -1

for i in range(len(result_rber_pots.communities)):
    for j in result_rber_pots.communities[i]:
        nodes1.loc[j,'community_rber_pots'] = i


# ## Rappresentazione grafica risultati

# Si rappresentano con colori diversi community diverse e si sovrappongono i confini territoriali dei quartieri in modo da verificare graficamente i risultati

# ### Spinglass

# In[33]:


quartieri_directory = os.path.join(current_dir, 'Quartieri')
quartieri_napoli = gpd.read_file(quartieri_directory)
n_quartieri = len(quartieri_napoli)

colors = [(random(),random(),random(),1) for i in range(len(nodes1['community_spinglass'].unique()))]
cc = [colors[i] for i in list(nodes1['community_spinglass'])]

fig, ax = ox.plot_graph(g1, node_color=cc, node_edgecolor=cc,
                        fig_height=11, fig_width=11,
                        show=False, close=False,
                        node_size=4, node_zorder=2, edge_linewidth=0.4,
                       margin=0)

for i in range(n_quartieri):
    polygon = quartieri_napoli['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    polypatch = PolygonPatch(polygon, alpha=1, zorder=2, color = 'black', fill = False)
    ax.add_patch(polypatch)
    
plt.title('Spinglass', {'fontsize': 23})
fig_file_name = os.path.join(picture_dir, 'splinglass')
plt.savefig(fig_file_name,bbox_inches='tight')


# ### Eigenvector

# In[34]:


quartieri_directory = os.path.join(current_dir, 'Quartieri')
quartieri_napoli = gpd.read_file(quartieri_directory)
n_quartieri = len(quartieri_napoli)

colors = [(random(),random(),random(),1) for i in range(len(nodes1['community_eigenvector'].unique()))]
cc = [colors[i] for i in list(nodes1['community_eigenvector'])]

fig, ax = ox.plot_graph(g1, node_color=cc, node_edgecolor=cc,
                        fig_height=11, fig_width=11,
                        show=False, close=False,
                        node_size=4, node_zorder=2, edge_linewidth=0.4,
                       margin=0)

for i in range(n_quartieri):
    polygon = quartieri_napoli['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    polypatch = PolygonPatch(polygon, alpha=1, zorder=2, color = 'black', fill = False)
    ax.add_patch(polypatch)
    
plt.title('Eigenvector', {'fontsize': 23})
fig_file_name = os.path.join(picture_dir, 'eigenvector')
plt.savefig(fig_file_name,bbox_inches='tight')


# ### Leiden

# In[35]:


quartieri_directory = os.path.join(current_dir, 'Quartieri')
quartieri_napoli = gpd.read_file(quartieri_directory)
n_quartieri = len(quartieri_napoli)

colors = [(random(),random(),random(),1) for i in range(len(nodes1['community_leiden'].unique()))]
cc = [colors[i] for i in list(nodes1['community_leiden'])]

fig, ax = ox.plot_graph(g1, node_color=cc, node_edgecolor=cc,
                        fig_height=11, fig_width=11,
                        show=False, close=False,
                        node_size=4, node_zorder=2, edge_linewidth=0.4,
                       margin=0)

for i in range(n_quartieri):
    polygon = quartieri_napoli['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    polypatch = PolygonPatch(polygon, alpha=1, zorder=2, color = 'black', fill = False)
    ax.add_patch(polypatch)
    
plt.title('Leiden', {'fontsize': 23})
fig_file_name = os.path.join(picture_dir, 'leiden')
plt.savefig(fig_file_name,bbox_inches='tight')


# ### CPM

# In[36]:


quartieri_directory = os.path.join(current_dir, 'Quartieri')
quartieri_napoli = gpd.read_file(quartieri_directory)
n_quartieri = len(quartieri_napoli)

colors = [(random(),random(),random(),1) for i in range(len(nodes1['community_cpm'].unique()))]
cc = [colors[i] for i in list(nodes1['community_cpm'])]

fig, ax = ox.plot_graph(g1, node_color=cc, node_edgecolor=cc,
                        fig_height=11, fig_width=11,
                        show=False, close=False,
                        node_size=4, node_zorder=2, edge_linewidth=0.4,
                       margin=0)

for i in range(n_quartieri):
    polygon = quartieri_napoli['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    polypatch = PolygonPatch(polygon, alpha=1, zorder=2, color = 'black', fill = False)
    ax.add_patch(polypatch)
    
plt.title('Cpm', {'fontsize': 23})
fig_file_name = os.path.join(picture_dir, 'cpm')
plt.savefig(fig_file_name,bbox_inches='tight')


# ### Rber Pots

# In[37]:


quartieri_directory = os.path.join(current_dir, 'Quartieri')
quartieri_napoli = gpd.read_file(quartieri_directory)
n_quartieri = len(quartieri_napoli)

colors = [(random(),random(),random(),1) for i in range(len(nodes1['community_rber_pots'].unique()))]
cc = [colors[i] for i in list(nodes1['community_rber_pots'])]

fig, ax = ox.plot_graph(g1, node_color=cc, node_edgecolor=cc,
                        fig_height=11, fig_width=11,
                        show=False, close=False,
                        node_size=4, node_zorder=2, edge_linewidth=0.4,
                       margin=0)

for i in range(n_quartieri):
    polygon = quartieri_napoli['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    polypatch = PolygonPatch(polygon, alpha=1, zorder=2, color = 'black', fill = False)
    ax.add_patch(polypatch)
    
plt.title('Rber Pots', {'fontsize': 23})
fig_file_name = os.path.join(picture_dir, 'rber_pots')
plt.savefig(fig_file_name,bbox_inches='tight')


# ## Evaluation

# In[38]:


community_columns = ['community_spinglass','community_eigenvector','community_leiden','community_cpm','community_rber_pots']
for community_column in community_columns:
    accuracy = accuracy_comunity(community_column)
    print('Accuracy of',community_column[10:],'algorithm:',accuracy)

