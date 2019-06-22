#!/usr/bin/env python
# coding: utf-8

# # Resilience

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from copy import deepcopy

current_dir = os.getcwd()
picture_dir = os.path.join(current_dir, 'grafici')
objects_dir = os.path.join(current_dir, 'objects')
network_file_name = os.path.join(objects_dir, 'napoli.network')

with open(network_file_name, 'rb') as napoli_network_file:
    g = pickle.load(napoli_network_file)


# In[2]:


print('Number of strongly directed components:',nx.number_strongly_connected_components(g))
print('Number of weakly directed components:',nx.number_weakly_connected_components(g))


# Ricerca dei punti che non appartengono alla giant component

# In[3]:


comps = list(nx.strongly_connected_components(g))
comps.sort(key=len, reverse=True)
not_connected_points = set()
for comp in comps[1:]:
    not_connected_points = not_connected_points.union(comp)


# Rappresentazione grafica dei nodi che non appartengono alla giant component

# In[4]:


fig, ax = ox.plot_graph(g, fig_height=13, fig_width=13, 
                        show=False, close=False, 
                        edge_color='black', axis_off = False)

for node in not_connected_points:
    ax.scatter(g.node[node]['x'],
               g.node[node]['y'], 
               c='red', s=13)


# Si scarica la rete pedonale di Napoli per verificare quante strongly connected components. C'è una sola componente e quindi si deduce che i nodi non appartenenti alla giant component sono in corrispondenza di zone pedonali.

# In[5]:


g_walk = ox.graph_from_place('Napoli, Italia', network_type="walk")

comps_walk = list(nx.strongly_connected_components(g_walk))

comps_walk.sort(key=len, reverse=True)
not_connected_points_walk = set()
for comp in comps_walk[1:]:
    not_connected_points_walk = not_connected_points_walk.union(comp)
    
print('Numero di componenti fortemente connesse:',len(comps_walk))


# ATTENZIONE: richiede qualche minuto
# 
# CONSIGLIO: passare alla cella successiva che carica i risultati da file

# In[7]:


def get_second_item(l):
    return l[1]

edge_betweenness = nx.edge_betweenness_centrality(g)
#si creano due copie della rete: ad una si tolgono i link per betweenness crescente e all'altra decrescente
g_copy_asc = deepcopy(g)
g_copy_desc = deepcopy(g)

#estrazione e ordinamento delle componenti connesse per numero di nodi
components = list(nx.strongly_connected_components(g_copy_asc))
components.sort(key=len, reverse=True)

#calcolo della frazione della giant component
frac_connected_component_asc = [len(components[0])/g_copy_asc.number_of_nodes()]
frac_connected_component_desc = [len(components[0])/g_copy_desc.number_of_nodes()]

#ordinamento link per edge betweenness
betweenness_asc = list(edge_betweenness.items())
betweenness_asc.sort(key=get_second_item)
betweenness_desc = list(edge_betweenness.items())
betweenness_desc.sort(key=get_second_item, reverse=True)

#rimozione graduale dei linke per edge betweenness crescente
for edge_to_remove_asc in betweenness_asc:
    g_copy_asc.remove_edge(*edge_to_remove_asc[0])
    components = list(nx.strongly_connected_components(g_copy_asc))
    components.sort(key=len, reverse=True)
    frac_connected_component_asc.append(len(components[0])/g_copy_asc.number_of_nodes())
    
#rimozione graduale dei linke per edge betweenness decrescente
for edge_to_remove_desc in betweenness_desc:
    g_copy_desc.remove_edge(*edge_to_remove_desc[0])
    components = list(nx.strongly_connected_components(g_copy_desc))
    components.sort(key=len, reverse=True)
    frac_connected_component_desc.append(len(components[0])/g_copy_desc.number_of_nodes())

#salvataggio dei dati su file binario
frac_connected_component_asc_file_name_bet = os.path.join(objects_dir, 'frac_connected_component_asc_betweennes.list')
frac_connected_component_desc_file_name_bet = os.path.join(objects_dir, 'frac_connected_component_desc_betweennes.list')
with open(frac_connected_component_asc_file_name_bet, 'wb') as frac_connected_component_asc_file:
    pickle.dump(frac_connected_component_asc, frac_connected_component_asc_file)
    
with open(frac_connected_component_desc_file_name_bet, 'wb') as frac_connected_component_desc_file:
    pickle.dump(frac_connected_component_desc, frac_connected_component_desc_file)


# Caricamento dei risultati da file

# In[6]:


#caricamento dei dati da file binario
frac_connected_component_asc_file_name_bet = os.path.join(objects_dir, 'frac_connected_component_asc_betweennes.list')
frac_connected_component_desc_file_name_bet = os.path.join(objects_dir, 'frac_connected_component_desc_betweennes.list')
with open(frac_connected_component_asc_file_name_bet, 'rb') as frac_connected_component_asc_file:
    frac_connected_component_asc_bet = pickle.load(frac_connected_component_asc_file)
    
with open(frac_connected_component_desc_file_name_bet, 'rb') as frac_connected_component_desc_file:
    frac_connected_component_desc_bet = pickle.load(frac_connected_component_desc_file)

x = [i for i in range(len(frac_connected_component_asc_bet))]

plt.figure(figsize=(16,12))
plt.plot(x,frac_connected_component_asc_bet, color='r',label='Ascendent')
plt.plot(x,frac_connected_component_desc_bet, color='b',label='Descendent')
plt.legend()
plt.xlabel("Edges removed")
plt.ylabel("Fraction of giant component")
plt.title('Betweenness resilience', {'fontsize': 23})

fig_file_name = os.path.join(picture_dir, 'giant_comp_betweenness')
plt.savefig(fig_file_name,bbox_inches='tight')
plt.show()


# Definizione della misura di overlap

# In[7]:


def overlap_coeff(g, edge):
    node_origin = edge[0]
    node_dest = edge[1]
    neighbors_origin = set([i for i in g.neighbors(node_origin)])
    neighbors_dest = set([i for i in g.neighbors(node_dest)])
    union = neighbors_origin.union(neighbors_dest)
    intersection = neighbors_origin.intersection(neighbors_dest)
    overlap = len(intersection)/len(union)
    return overlap

def get_second_item(l):
    return l[1]


# Si ripetono gli stessi passaggi precedenti ma utilizzando la misura di overlap.
# 
# ATTENZIONE: richiede qualche minuto
# 
# CONSIGLIO: passare alla cella successiva che carica i risultati da file

# In[13]:


overlap = dict()
for edge in g.edges():
    overlap[edge] = overlap_coeff(g, edge)

g_copy_asc = deepcopy(g)
g_copy_desc = deepcopy(g)
components = list(nx.strongly_connected_components(g_copy_asc))
components.sort(key=len, reverse=True)
frac_connected_component_asc = [len(components[0])/g_copy_asc.number_of_nodes()]
frac_connected_component_desc = [len(components[0])/g_copy_desc.number_of_nodes()]
overlap_asc = list(overlap.items())
overlap_asc.sort(key=get_second_item)
overlap_desc = list(overlap.items())
overlap_desc.sort(key=get_second_item, reverse=True)

for edge_to_remove_asc in overlap_asc:
    g_copy_asc.remove_edge(*edge_to_remove_asc[0])
    components = list(nx.strongly_connected_components(g_copy_asc))
    components.sort(key=len, reverse=True)
    frac_connected_component_asc.append(len(components[0])/g_copy_asc.number_of_nodes())
    
for edge_to_remove_desc in overlap_desc:
    g_copy_desc.remove_edge(*edge_to_remove_desc[0])
    components = list(nx.strongly_connected_components(g_copy_desc))
    components.sort(key=len, reverse=True)
    frac_connected_component_desc.append(len(components[0])/g_copy_desc.number_of_nodes())
    
frac_connected_component_asc_file_name_overlap = os.path.join(objects_dir, 'frac_connected_component_asc_overlap.list')
frac_connected_component_desc_file_name_overlap = os.path.join(objects_dir, 'frac_connected_component_desc_overlap.list')
with open(frac_connected_component_asc_file_name_overlap, 'wb') as frac_connected_component_asc_file:
    pickle.dump(frac_connected_component_asc, frac_connected_component_asc_file)
    
with open(frac_connected_component_desc_file_name_overlap, 'wb') as frac_connected_component_desc_file:
    pickle.dump(frac_connected_component_desc, frac_connected_component_desc_file)


# Caricamento dei risultati da file

# In[8]:


frac_connected_component_asc_file_name_overlap = os.path.join(objects_dir, 'frac_connected_component_asc_overlap.list')
frac_connected_component_desc_file_name_overlap = os.path.join(objects_dir, 'frac_connected_component_desc_overlap.list')
with open(frac_connected_component_asc_file_name_overlap, 'rb') as frac_connected_component_asc_file:
    frac_connected_component_asc_overlap = pickle.load(frac_connected_component_asc_file)
    
with open(frac_connected_component_desc_file_name_overlap, 'rb') as frac_connected_component_desc_file:
    frac_connected_component_desc_overlap = pickle.load(frac_connected_component_desc_file)

x = [i for i in range(len(frac_connected_component_asc_overlap))]

plt.figure(figsize=[16,12])
plt.plot(x,frac_connected_component_asc_overlap, color='r',label='Ascendent')
plt.plot(x,frac_connected_component_desc_overlap, color='b',label='Descendent')
plt.legend()
plt.xlabel("Edges removed")
plt.ylabel("Fraction of giant component")
plt.title('Overlap resilience', {'fontsize': 23})

fig_file_name = os.path.join(picture_dir, 'giant_comp_overlap')
plt.savefig(fig_file_name,bbox_inches='tight')
plt.show()

