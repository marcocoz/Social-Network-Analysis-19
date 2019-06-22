#!/usr/bin/env python
# coding: utf-8

# # Data Collection

# Estraggo la mappa di Napoli e la rappresento graficamente

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import folium
import pickle
import geopandas as gpd
import matplotlib.cm as cm
import shapefile as shp
from descartes.patch import PolygonPatch
import os
from matplotlib.lines import Line2D
import shapely.wkt

#estrazione della directory corrente
current_dir = os.getcwd()
picture_dir = os.path.join(current_dir, 'grafici')

#estrazione della rete di Napoli
g = ox.graph_from_place('Napoli, Italia', network_type="drive")
g_projected = ox.project_graph(g)

#rappresentazione grafica della rete 
fig, ax = ox.plot_graph(g, fig_height=10, fig_width=10, 
                        show=False, close=False, 
                        edge_color='black')

plt.title('Rete stradale di Napoli', {'fontsize': 23})
fig_file_name = os.path.join(picture_dir, 'rete_napoli')
plt.savefig(fig_file_name,bbox_inches='tight')

#assegnazione nodi e link della rete alle variabili nodes ed edges
nodes, edges = ox.graph_to_gdfs(g)
nodes.head()


# Visto che abbiamo un multigraph si rappresentano sulla mappa le coppie di nodi che presentano più link nella stessa direzione

# In[2]:


double_edges = edges.groupby(['u','v']).aggregate('geometry').count()
double_edges = double_edges[double_edges>1]
double_edges = list(double_edges.index)
edges_list = [(i[0],i[1]) for i in list(edges[['u','v']].values)]
colors = ['blue', 'red']
color_list = [('blue', 'red')[i in double_edges] for i in edges_list]

fig, ax = ox.plot_graph(g, fig_height=13, fig_width=13, 
                        show=False, close=False, 
                        edge_color=color_list)

fig_file_name = os.path.join(picture_dir, 'rete_napoli_double_edges')
plt.savefig(fig_file_name,bbox_inches='tight')

plt.show()


# ## Ospedali

# #### Ospedali da fscrape

# - Lettura delle coordinate degli ospedali da un file csv
# - Ricerca del nodo più vicino ad ogni ospedale all'interno del network

# In[3]:


tree = KDTree(nodes[['y', 'x']], metric='euclidean')

osp_file_name = os.path.join(current_dir, 'new_ospedali_Napoli_20190416.csv')
df = pd.read_csv(osp_file_name, sep=';')

coord_ospedali = df['COORDINATE MAPPA']
n_ospedali = len(coord_ospedali)

coord_osp = list()
#indici del KDTree
osp_idx = list()
#indici dei nodi del grafo g
closest_node_to_osp = list()
for i in range(n_ospedali):
    coord_osp.append(coord_ospedali[i].split())
    osp_idx.append(tree.query([coord_osp[i]], k=1, return_distance=False)[0])
    closest_node_to_osp.append(nodes.iloc[osp_idx[i]].index.values[0])


# Si aggiunge l'attributo ospedale ai nodi

# In[4]:


"""si creano due dizionari per i nuovi attributi ospedale e ID_ospedale. Attraverso questi
si valorizzeranno i nuovi attributi"""
nodes_ospedali = dict()
ID_nodes_ospedali = dict()

#si creano altri due dizionari per valorizzare i dati di default: '' per ospedale e -1 per ID 
default_ospedale = {i:'' for i in set(g.nodes())}
default_ID_ospedale = {i:-1 for i in set(g.nodes())}
#si attribuiscono i valori di default ai nodi
nx.set_node_attributes(g, default_ospedale, 'ospedale')
nx.set_node_attributes(g, default_ID_ospedale, 'ID_ospedale')

#si assegna la descrizione dell'ospedale e l'ID all'interno del dizionario dei nodi relativi agli ospedali
for i in range(n_ospedali):
        nodes_ospedali[closest_node_to_osp[i]] = df.loc[i,'RAGIONE SOCIALE']
        ID_nodes_ospedali[closest_node_to_osp[i]] = i
        
#si attribuiscono i valori dei dizionari ai nuovi attributi dei nodi della rete
nx.set_node_attributes(g, nodes_ospedali, 'ospedale')
nx.set_node_attributes(g, ID_nodes_ospedali, 'ID_ospedale')

nodes, edges = ox.graph_to_gdfs(g)
nodes.sort_values('ID_ospedale', ascending = False).head()


# Si rappresenta nella mappa in rosso gli ospedali e in verde i nodi più vicini ad essi nella mappa

# In[5]:


from matplotlib.lines import Line2D

#rappresentazione grafico del network di Napoli
fig, ax = ox.plot_graph(g, fig_height=10, fig_width=10, 
                        show=False, close=False, 
                        edge_color='black', axis_off = False)

legend_elements = [Line2D([0], [0], marker='o', color='r', label='Ospedale', linestyle=''),
                  Line2D([0], [0], marker='o', color='g', label='Nodo più vicino', linestyle='')]

plt.legend(handles=legend_elements)
plt.title('Rete stradale di Napoli con ospedali', {'fontsize': 23})

for i in range(n_ospedali):
    #rappresentazione degli ospedali in rosso
    ax.scatter(float(coord_osp[i][1]), float(coord_osp[i][0]), c='red', s=13)
    #rappresentazione dei nodi più vicini agli ospedali in verde
    ax.scatter(g.node[closest_node_to_osp[i]]['x'],
               g.node[closest_node_to_osp[i]]['y'], 
               c='green', s=13)
    
fig_file_name = os.path.join(picture_dir, 'rete_napoli_osp')
plt.savefig(fig_file_name,bbox_inches='tight')


# Stessa rappresentazione di prima ma su una mappa interattiva

# In[6]:


start_point=[float(coord_osp[0][0]),float(coord_osp[0][1])]
m = folium.Map(
    location=start_point,
    #tiles='Stamen Toner',
    #tiles='Stamen Terrain',
    zoom_start=12
)

descrizione_osp = list()

for i in range(len(df)):
    descrizione_osp.append(df.iloc[i,0]+'.\n'+df.iloc[i,1]+'.\n'+df.iloc[i,2]+'\n'+
                           str(df.iloc[i,3])+'\n'+df.iloc[i,4]+' '+df.iloc[i,5])

for i in range(n_ospedali):
    folium.Marker(
        location=[float(coord_osp[i][0]),float(coord_osp[i][1])],
        popup='Ospedale '+str(i)+'.\n'+descrizione_osp[i],
        icon=folium.Icon(color='red')
    ).add_to(m)
    
for i in range(n_ospedali):
    folium.Marker(
        location=[g.node[closest_node_to_osp[i]]['y'],g.node[closest_node_to_osp[i]]['x']],
        popup='Nodo più vicino all\'ospedale '+str(i),
        icon=folium.Icon(color='green')
    ).add_to(m)
    
    
m


# #### Ospedali sito del comune

# I dati comprendono i lotti degli ospedali, delle aziende ospedaliere universitarie (primo e secondo  policlinico) e delle aziende sanitarie di interesse nazionale nel comune di Napoli. Il dato è costruito utilizzando i database catastali per meglio definire i lotti sul territorio ed è stato realizzato dal Comune di Napoli.

# Lettura e rappresentazione grafica della mappa degli ospedali di Napoli

# In[7]:


ospedali_comune_dir = os.path.join(current_dir, 'ospedali_comune')
ospedali_comune_file_name = os.path.join(ospedali_comune_dir, 'ospedali_presidisanitari.shp')

sf = shp.Reader(ospedali_comune_file_name)

plt.figure()
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)


# Funzione per estrarre le coordinate dei bordi di un poligono come lista

# In[8]:


def get_coord_confini(polygon):
    confini_interm2 = ''
    confini_interm = str(polygon)[10:-2].split('), (')
    for i in confini_interm:
        confini_interm2 += ', '+i

    confini = list()
    confini_interm2 = confini_interm2[2:].split(', ')
    for i in confini_interm2:
        confini.append(i.split())

    confini = [[float(i[0]), float(i[1])] for i in confini]
    confini = [[confini[i][1],confini[i][0]] for i in range(len(confini))]
    return confini


# Estrazione dei dati relativi agli ospedali dal file fornito dal comune di Napoli

# In[9]:


ospedali_comune_dir = os.path.join(current_dir, 'ospedali_comune')
ospedali = gpd.read_file(ospedali_comune_dir)
n_ospedali_comune = len(ospedali)


# Estrazione delle coordinate dei punti che formano i confini dei poligoni relativi agli ospedali. Per ogni punto trovato ricerca del nodo rella rete più vicino ad esso

# In[10]:


"""Si crea un oggetto KDTree all'interno del quale si inseriscono tutte le coordinate dei nodi della
rete. Questo serve per trovare il nodo più vicino ad un punto qualunque sulla mappa.
Grazie a questo oggetto si trovano i nodi più vicini ai confini delle zone ospedaliere"""

tree = KDTree(nodes[['y', 'x']], metric='euclidean')
#si inizializza la lista che conterrà i nodi più vicini alle zone ospedaliere
closest_nodes = list()
for i in range(n_ospedali_comune):
    #indici degli ospedali
    osp_idx = list()
    closest_node_to_osp_comune = list()
    #poligono contenente l'ospedale
    polygon = ospedali['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    #si estraggono le coordinate del bordo del poligono
    coords_polygon = get_coord_confini(polygon)
    #si trovano i nodi più vicini all'ospedale e li inserisco infine nella variabile closest_nodes
    for j in range(len(coords_polygon)):
        osp_idx.append(tree.query([coords_polygon[j]], k=1, return_distance=False)[0])
        closest_node_to_osp_comune.append(nodes.iloc[osp_idx[j]].index.values[0])
    closest_nodes.append(closest_node_to_osp_comune)

#stampa del numero di nodi individuati per ogni ospedale
for i in range(n_ospedali_comune):
    nome_ospedale = ospedali.loc[i, 'DENOM']
    num_nodi_per_ospedale = len(set(closest_nodes[i]))
    print(num_nodi_per_ospedale, 'nodi individuati per', nome_ospedale)

#calcolo di quanti nodi ho individuato in totale
nodi_trovati = set()
for i in range(len(closest_nodes)):
    nodi_trovati = nodi_trovati.union(set(closest_nodes[i]))
    
print('\n')
print('In totale sono stati individuati',len(nodi_trovati),'nodi')

#eliminazione dei duplicati
closest_nodes = [set(i) for i in closest_nodes]


# Si aggiungono i nodi che stanno all'interno della zona ospedaliera nel seguente modo: per ogni zona ospedaliera:
# - si scarica la rete e si individuano i nodi
# - si trovano i nodi in comune con la rete dell'intera città
# - si assegna ad ogni nodo trovato il relativo ospedale

# In[11]:


for i in range(n_ospedali_comune):
    polygon = ospedali['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    try:
        #estrazione della rete delimitata dalla zona ospedaliera
        g_ospedale = ox.graph_from_polygon(polygon, network_type='drive_service', retain_all = True)
        #intersezione fra nodi dentro la zona ospedaliera e quelli della rete della città
        intersezione = set(g.nodes()).intersection(set(g_ospedale.nodes()))
        #numero di nodi prima di inserire quelli all'interno del poligonow
        n_nodi_osp = len(closest_nodes[i])
        for node in intersezione:
            closest_nodes[i].add(node)
        if len(closest_nodes[i])==n_nodi_osp:
            print('Non sono stati individuati nuovi nodi all\'interno della zona ospedaliera', i)
        else:
            print('Sono stati individuati', closest_nodes[i]-n_nodi_osp,'nuovi nodi')
    except:
        #se il poligono è troppo piccolo non riesce ad individuare una rete e va in errore
        #si gestisce considerando che in questo caso non ci sono nodi all'interno
        print('Non sono stati individuati nodi all\'interno della zona ospedaliera', i)


# In[12]:


"""si crea un dizionario dove si inseriranno i valori del nuovo attributo 'ID_ospedale_comune'
le chiavi del dizionario sono gli indici dei nodi della rete e i valori del dizionario sono
i valori cha si assegneranno al nuovo attributo"""
ospedali_comune = {i:[] for i in set(g.nodes())}
ospedali_comune = pd.Series(ospedali_comune)

#inserimento dei valori dei rispettivi ospedali ai nodi all'interno del dizionario
for i in range(len(closest_nodes)):
    for node in closest_nodes[i]:
        ospedali_comune[node].append(i)
        
#attribuzione dei valori del dizionario all'attributo ID_ospedale_comune
nx.set_node_attributes(g, ospedali_comune, 'ID_ospedale_comune')

#ricalcolo della variabile nodes contenente i dati sui nodi
nodes, edges = ox.graph_to_gdfs(g)
nodes.sort_values('ID_ospedale_comune', ascending=False).head()


# Si controlla se ci sono ospedali che hanno individuato gli stessi nodi

# In[13]:


#inizializzazione della variabile nodi_ospedali_ambigui che conterrà la lista dei nodi ambigui
nodi_ospedali_ambigui = list()
for i in range(len(nodes)):
    #si controlla quali nodi appartengono a più di un ospedale
    if len(nodes['ID_ospedale_comune'].iloc[i])>1:
        nodi_ospedali_ambigui.append(nodes.index[i])
        print('Nodo',nodes.index[i],'ospedali',nodes['ID_ospedale_comune'].iloc[i])


# Rappresentazione della rete di Napoli evidenziando in rosso i nodi relativi agli ospedali individuati. In verde i nodi ambigui.

# In[14]:


#rappresentazione grafico del network di Napoli
fig, ax = ox.plot_graph(g, fig_height=13, fig_width=13,
                        show=False, close=False,
                        edge_color='black', axis_off = False,
                        node_alpha=.6, edge_alpha=.2
                       )

#rappresentazione nodi relativi agli ospedali in rosso
for i in range(len(closest_nodes)):
    for j in range(len(closest_nodes[i])):
        ax.scatter(g.node[list(closest_nodes[i])[j]]['x'],
                   g.node[list(closest_nodes[i])[j]]['y'], 
                   c='r', s=7)
        
#rappresentazione nodi relativi a più di un ospedale
for i in range(len(nodi_ospedali_ambigui)):
    ax.scatter(g.node[nodi_ospedali_ambigui[i]]['x'],
               g.node[nodi_ospedali_ambigui[i]]['y'], 
               c='g', s=7)

plt.title('Rete stradale Napoli con ospedali dal sito del comune', {'fontsize': 23})
legend_elements = [Line2D([0], [0], marker='o', color='r', label='Nodo ospedale', linestyle=''),
                  Line2D([0], [0], marker='o', color='g', label='Nodo relativo a più di un ospedale', linestyle='')]

plt.legend(handles=legend_elements)
fig_file_name = os.path.join(picture_dir, 'rete_napoli_osp_comune')
plt.savefig(fig_file_name,bbox_inches='tight')
plt.show()


# Stessa rappresentazione ma con folium

# In[15]:


start_point=[g.node[list(closest_nodes[0])[0]]['y'],g.node[list(closest_nodes[0])[0]]['x']]
m = folium.Map(
    location=start_point,
    #tiles='Stamen Toner',
    #tiles='Stamen Terrain',
    zoom_start=12
)

for i in range(len(closest_nodes)):
    for j in range(len(closest_nodes[i])):
        node = list(closest_nodes[i])[j]
        if node not in nodi_ospedali_ambigui:
            folium.Marker(
                location=[g.node[node]['y'], g.node[node]['x']],
                popup='Ospedale '+str(nodes.loc[node,'ID_ospedale_comune']),
                icon=folium.Icon(color='red')
            ).add_to(m)
            
for i in range(len(nodi_ospedali_ambigui)):
    node = nodi_ospedali_ambigui[i]
    folium.Marker(
        location=[g.node[node]['y'], g.node[node]['x']],
        popup='Nodo '+str(node)+'\nOspedali '+str(nodes.loc[node,'ID_ospedale_comune']),
        icon=folium.Icon(color='green')
    ).add_to(m)
    
m


# Risoluzione i nodi disambigui. Si guarda nella mappa sopra a quali ospedali realmente corrispondono i nodi ambigui. Gli ospedali 6 e 7 sono lo stesso ospedale e vengono uniti automaticamente con il seguente procedimento. Analogamente per gli ospedali 9 e 10. In pratica l'ospedale 10 e l'ospedale 7 spariscono.

# In[16]:


nodes['ID_ospedale_comune_finale'] = -1
for i in range(len(nodes)):
    #si controlla quali nodi appartengono a un solo ospedale e si attribuisce il reltivo ospedale ai nodi
    try:
        if len(nodes['ID_ospedale_comune'].iloc[i])==1:
            nodes.loc[nodes.index[i], 'ID_ospedale_comune_finale'] = nodes['ID_ospedale_comune'].iloc[i][0]
    except:
        a=1

#si crea il dizionario che contiene i valori finali dell'attributo ID_ospedale_comune
ID_ospedale_comune_finale = dict(nodes['ID_ospedale_comune_finale'])

#si attribuiscono manualmente i nodi giusti agli ospedali trovati
ID_ospedale_comune_finale[2909394756] = 1
ID_ospedale_comune_finale[25366799] = 20
ID_ospedale_comune_finale[4612101349 ] = 20
ID_ospedale_comune_finale[4950661485] = 9
ID_ospedale_comune_finale[4950661457] = 9
ID_ospedale_comune_finale[418007760] = 9
ID_ospedale_comune_finale[418007762]=9
ID_ospedale_comune_finale[4947646364] = 9
ID_ospedale_comune_finale[418007909] = 9
ID_ospedale_comune_finale[324444158] = 8
ID_ospedale_comune_finale[925743197] = 8
ID_ospedale_comune_finale[4990140500] = 5
ID_ospedale_comune_finale[927154064] = 6
ID_ospedale_comune_finale[927154086] = 6
ID_ospedale_comune_finale[927153925]  = 6
ID_ospedale_comune_finale[925734709] = 5

#attribuzione dei valori del dizionario all'attributo ID_ospedale_comune
nx.set_node_attributes(g, ID_ospedale_comune_finale, 'ID_ospedale_comune')

#ricalcolo della variabile nodes contenente i dati sui nodi
nodes, edges = ox.graph_to_gdfs(g)
nodes.sort_values('ID_ospedale_comune',ascending=False).head()


# Rappresentazione contemporaneamente degli ospedali dal sito del comune con gli altri

# In[17]:


from matplotlib.lines import Line2D
fig, ax = ox.plot_graph(g, fig_height=10, fig_width=10, 
                        show=False, close=False, 
                        edge_color='b', edge_linewidth = .4, axis_off = False)

legend_elements = [Line2D([0], [0], marker='o', color='r', label='Ospedale', linestyle=''),
                  Line2D([0], [0], marker='o', color='g', label='Nodo più vicino', linestyle='')]

plt.legend(handles=legend_elements)
plt.title('Rete stradale di Napoli con ospedali', {'fontsize': 23})

for i in range(n_ospedali):
    ax.scatter(float(coord_osp[i][1]), float(coord_osp[i][0]), c='red', s=13)
    ax.scatter(g.node[closest_node_to_osp[i]]['x'],
               g.node[closest_node_to_osp[i]]['y'], 
               c='black', s=13)

for i in range(n_ospedali_comune):
    polygon = ospedali['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    polypatch = PolygonPatch(polygon, alpha=0.6, zorder=2, color = 'g', fill = True)
    ax.add_patch(polypatch)


# ## Quartieri

# Lettura e rappresentazione grafica della mappa dei quartieri di Napoli

# In[3]:


quartieri_directory = os.path.join(current_dir, 'Quartieri')
quartieri_file_name = os.path.join(quartieri_directory, 'quartieri2001.shp')

sf = shp.Reader(quartieri_file_name)

plt.figure()
for shape in sf.shapeRecords():
    x = [i[0] for i in shape.shape.points[:]]
    y = [i[1] for i in shape.shape.points[:]]
    plt.plot(x,y)


# In[4]:


quartieri_napoli = gpd.read_file(quartieri_directory)
quartieri_napoli['QUART'].tolist()


# Rappresento graficamente il network di ogni quartiere.
# 
# ATTENZIONE: richiede qualche minuto!

# In[ ]:


quartieri_napoli = gpd.read_file(quartieri_directory)
n_quartieri = len(quartieri_napoli) #30 quartieri
for j in range(n_quartieri):
    #estrazione del poligono relativo al quartiere
    polygon = quartieri_napoli['geometry'].iloc[j]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    #estrazione della rete delimitata dai confini del quartiere j
    G = ox.graph_from_polygon(polygon, network_type='drive_service')
    G_projected = ox.project_graph(G)
    #stampa del nome del quartiere
    print(quartieri_napoli.loc[j,'QUART'])
    #rappresentazione grafica della rete del quartiere
    fig, ax = ox.plot_graph(G)


# Si individua il quartiere dei nodi della rete della città di Napoli nel seguente modo: per ogni quartiere
# - si scarica la rete e si individuano i nodi
# - si trovano i nodi in comune con la rete dell'intera città
# - si assegna ad ogni nodo della rete della città il proprio quartiere

# In[5]:


"""Come per gli ospedali si aggiungono due variabili, quartiere e ID_quartiere e per farlo si creano
i relativi dizionari che popoleranno i nuovi attributi dei nodi"""

quartieri_napoli = gpd.read_file(quartieri_directory)
n_quartieri = len(quartieri_napoli) #30 quartieri
nodes_quartier = dict()
ID_nodes_quartier = dict()

for i in range(n_quartieri):
    #estrazione del poligono relativo al quartiere
    polygon = quartieri_napoli['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    g_quartier = ox.graph_from_polygon(polygon, network_type='drive_service', retain_all = True)
    #intersezione fra nodi dentro il quartiere e quelli della rete della città
    intersezione = set(g.nodes()).intersection(set(g_quartier.nodes()))
    for j in list(intersezione):
        #assegnazione del valore del quartiere al dizionario
        nodes_quartier[j] = quartieri_napoli.loc[i,'QUART']
        #assegnazione del valore dell'ID del quartiere al dizionario
        ID_nodes_quartier[j] = i
    
print('Numero di nodi di cui è stato individuato il quartiere:', len(nodes_quartier))
print('Numero di nodi totali:', len(nodes))

#assegnazione del valore 'Quartiere non identificato' come valore di default
default_quartier = {i:'Quartiere non identificato' for i in set(g.nodes())}
default_ID_quartier = {i:-1 for i in set(g.nodes())}
nx.set_node_attributes(g, default_quartier, 'quartiere')
nx.set_node_attributes(g, default_ID_quartier, 'ID_quartiere')

#assegnazione del quartiere corrispondente ad ogni nodo
nx.set_node_attributes(g, nodes_quartier, 'quartiere')
nx.set_node_attributes(g, ID_nodes_quartier, 'ID_quartiere')

nodes, edges = ox.graph_to_gdfs(g)
nodes.head()


# Controllo che ogni nodo appartenga ad un solo quartiere.
# 
# ATTENZIONE: richiede qualche minuto!
# 
# CONSIGLIO: saltare la prossima cella a meno che non si vogliano verificare i risultati. 

# In[6]:


nodi_quartiere = list()

for i in range(n_quartieri):
    polygon = quartieri_napoli['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    g_quartier = ox.graph_from_polygon(polygon, network_type='drive_service', simplify=False)
    nodi_quartiere.append(set(g_quartier.nodes()))
    
flag_nodi_comuni = 0
for nodi0 in nodi_quartiere:
    for nodi1 in nodi_quartiere:
        inters = nodi0.intersection(nodi1)
        if len(inters) > 0 and len(inters)!= len(nodi0):
            flag_nodi_comuni += 1

if flag_nodi_comuni > 0:
    print("Ci sono nodi in comune")
else:
    print('Non ci sono nodi in comune')


# Rappresentazione della rete di Napoli evidenziando i quartieri

# In[22]:


from random import random

colors = [(random(),random(),random(),1) for i in range(len(quartieri_napoli)+1)]
cc = [colors[i] for i in list(nodes.ID_quartiere)]

fig, ax = ox.plot_graph(g, node_color=cc, node_edgecolor=cc,
                        fig_height=15, fig_width=15,
                        show=False, close=False,
                        node_size=11, node_zorder=2, edge_linewidth=0.4,
                       margin=0)

#si disegnano sopra i confini del quartiere presi dal file .shp
for i in range(n_quartieri):
    polygon = quartieri_napoli['geometry'].iloc[i]
    polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    polypatch = PolygonPatch(polygon, alpha=1, zorder=2, color = 'black', fill = False)
    ax.add_patch(polypatch)
    
legend_elements = [Line2D([0], [0], marker='o', color=colors[i], label=quartieri_napoli.loc[i,'QUART'], linestyle='') for i in range(len(quartieri_napoli))]
fig.legend(handles=legend_elements, loc=1, ncol=2)

fig_file_name = os.path.join(picture_dir, 'rete_napoli_quartieri')
plt.savefig(fig_file_name,bbox_inches='tight')

plt.show()


# In[23]:


x = Line2D([0], [0], marker='o', color=colors[i], label=quartieri_napoli.loc[i,'QUART'], linestyle='')
x.get_label()

def label(x):
    return x.get_label()
legend_elements.sort(key=label)


# ## Castelli

# In[24]:


castle_directory = os.path.join('Architettura_fortificata')
castle = gpd.read_file(castle_directory)
castle = castle[(castle['tipologia']=='castello')&(castle['nome'] != 'Museo Etnopreistoria del CAI di Napoli "Alfonso Piciocchi"')]
coord_castle = castle['geometry'].astype(dtype='str', copy=True)
n_castle = len(coord_castle)

coord_cas = list()
#indici del KDTree
castle_idx = list()
#indici dei nodi del grafo g
closest_node_to_castle = list()
for i in range(n_castle):
    coord_cas.append([coord_castle[i][7:-1].split()[1], coord_castle[i][7:-1].split()[0]])
    castle_idx.append(tree.query([coord_cas[i]], k=1, return_distance=False)[0])
    closest_node_to_castle.append(nodes.iloc[castle_idx[i]].index.values[0])


# In[25]:


"""si creano due dizionari per i nuovi attributi castle e ID_castle. Attraverso questi
si valorizzeranno i nuovi attributi"""
nodes_castles = dict()
ID_nodes_castles = dict()

#si creano altri due dizionari per valorizzare i dati di default: '' per castle e -1 per ID 
default_castle = {i:'' for i in set(g.nodes())}
default_ID_castle = {i:-1 for i in set(g.nodes())}
#si attribuiscono i valori di default ai nodi
nx.set_node_attributes(g, default_castle, 'castle')
nx.set_node_attributes(g, default_ID_castle, 'ID_castle')

#si assegna la descrizione dell'castle e l'ID all'interno del dizionario dei nodi relativi agli castles
for i in range(n_castle):
    nodes_castles[closest_node_to_castle[i]] = castle.loc[i,'nome']
    ID_nodes_castles[closest_node_to_castle[i]] = i
        
#si attribuiscono i valori dei dizionari ai nuovi attributi dei nodi della rete
nx.set_node_attributes(g, nodes_castles, 'castle')
nx.set_node_attributes(g, ID_nodes_castles, 'ID_castle')

nodes, edges = ox.graph_to_gdfs(g)
nodes.sort_values('ID_castle', ascending = False).head()


# ## Parchi

# In[26]:


parchi_file_name = os.path.join(current_dir, 'parco_csv.csv')
parchi = pd.read_csv(parchi_file_name, sep=';')

coord_parchi = parchi['Coordinate']
n_parchi = len(coord_parchi)

coord_par = list()
#indici del KDTree
par_idx = list()
#indici dei nodi del grafo g
closest_node_to_par = list()
for i in range(n_parchi):
    coord_par.append(coord_parchi[i].split(','))
    par_idx.append(tree.query([coord_par[i]], k=1, return_distance=False)[0])
    closest_node_to_par.append(nodes.iloc[par_idx[i]].index.values[0])


# In[27]:


"""si creano due dizionari per i nuovi attributi parco e ID_parco. Attraverso questi
si valorizzeranno i nuovi attributi"""
nodes_parchi = dict()
ID_nodes_parchi = dict()

#si creano altri due dizionari per valorizzare i dati di default: '' per parco e -1 per ID 
default_parco = {i:'' for i in set(g.nodes())}
default_ID_parco = {i:-1 for i in set(g.nodes())}
#si attribuiscono i valori di default ai nodi
nx.set_node_attributes(g, default_parco, 'parco')
nx.set_node_attributes(g, default_ID_parco, 'ID_parco')

#si assegna la descrizione dell'parco e l'ID all'interno del dizionario dei nodi relativi agli parchi
for i in range(n_parchi):
        nodes_parchi[closest_node_to_par[i]] = parchi.loc[i,'Denominazione']
        ID_nodes_parchi[closest_node_to_par[i]] = i
        
#si attribuiscono i valori dei dizionari ai nuovi attributi dei nodi della rete
nx.set_node_attributes(g, nodes_parchi, 'parco')
nx.set_node_attributes(g, ID_nodes_parchi, 'ID_parco')

nodes, edges = ox.graph_to_gdfs(g)
nodes.sort_values('ID_parco', ascending = False).head()


# ## Scuole

# In[28]:


scuole_file_name = os.path.join(current_dir, 'scuole_ok.csv')
scuole = pd.read_csv(scuole_file_name, sep=';')

coord_scuole = scuole['COORDINATE']
n_scuole = len(coord_scuole)

coord_sc = list()
#indici del KDTree
sc_idx = list()
#indici dei nodi del grafo g
closest_node_to_sc = list()
for i in range(n_scuole):
    coord_sc.append(coord_scuole[i].split(','))
    sc_idx.append(tree.query([coord_sc[i]], k=1, return_distance=False)[0])
    closest_node_to_sc.append(nodes.iloc[sc_idx[i]].index.values[0])


# In[29]:


"""si creano due dizionari per i nuovi attributi scuola e ID_scuola. Attraverso questi
si valorizzeranno i nuovi attributi"""
nodes_scuole = dict()
ID_nodes_scuole = dict()

#si creano altri due dizionari per valorizzare i dati di default: '' per scuola e -1 per ID 
default_scuola = {i:'' for i in set(g.nodes())}
default_ID_scuola = {i:-1 for i in set(g.nodes())}
#si attribuiscono i valori di default ai nodi
nx.set_node_attributes(g, default_scuola, 'scuola')
nx.set_node_attributes(g, default_ID_scuola, 'ID_scuola')

#si assegna la descrizione dell'scuola e l'ID all'interno del dizionario dei nodi relativi agli scuole
for i in range(n_scuole):
        nodes_scuole[closest_node_to_sc[i]] = scuole.loc[i,'SCUOLE']
        ID_nodes_scuole[closest_node_to_sc[i]] = i
        
#si attribuiscono i valori dei dizionari ai nuovi attributi dei nodi della rete
nx.set_node_attributes(g, nodes_scuole, 'scuola')
nx.set_node_attributes(g, ID_nodes_scuole, 'ID_scuola')

nodes, edges = ox.graph_to_gdfs(g)
nodes.sort_values('ID_scuola', ascending = False).head()


# ## Università

# In[30]:


universita_file_name = os.path.join(current_dir, 'universita_ok.csv')
universita = pd.read_csv(universita_file_name, sep=';')

coord_universita = universita['Coordinate']
n_universita = len(coord_universita)

coord_uni = list()
#indici del KDTree
uni_idx = list()
#indici dei nodi del grafo g
closest_node_to_uni = list()
for i in range(n_universita):
    coord_uni.append(coord_universita[i].split(','))
    uni_idx.append(tree.query([coord_uni[i]], k=1, return_distance=False)[0])
    closest_node_to_uni.append(nodes.iloc[uni_idx[i]].index.values[0])


# In[31]:


"""si creano due dizionari per i nuovi attributi universita e ID_universita. Attraverso questi
si valorizzeranno i nuovi attributi"""
nodes_universita = dict()
ID_nodes_universita = dict()

#si creano altri due dizionari per valorizzare i dati di default: '' per universita e -1 per ID 
default_universita = {i:'' for i in set(g.nodes())}
default_ID_universita = {i:-1 for i in set(g.nodes())}
#si attribuiscono i valori di default ai nodi
nx.set_node_attributes(g, default_universita, 'universita')
nx.set_node_attributes(g, default_ID_universita, 'ID_universita')

#si assegna la descrizione dell'universita e l'ID all'interno del dizionario dei nodi relativi agli universita
for i in range(n_universita):
        nodes_universita[closest_node_to_uni[i]] = universita.loc[i,'Università']
        ID_nodes_universita[closest_node_to_uni[i]] = i
        
#si attribuiscono i valori dei dizionari ai nuovi attributi dei nodi della rete
nx.set_node_attributes(g, nodes_universita, 'universita')
nx.set_node_attributes(g, ID_nodes_universita, 'ID_universita')

nodes, edges = ox.graph_to_gdfs(g)
nodes.sort_values('ID_universita', ascending = False).head()


# ## Zone Catastali

# Si individua la zona catastale dei nodi della rete della città di Napoli tramite lo stesso procedimento utilizzato per i quartieri

# In[32]:


file_name = os.path.join(current_dir, 'napoli_omi.csv')
zone = gpd.read_file(file_name)
n_zone = len(zone)
nodes_zona = dict()
ID_nodes_zona = dict()

for i in range(n_zone):
    #estrazione del poligono relativo alla zona catastale
    pol = zone.loc[i,'WKT']
    polygon = shapely.wkt.loads(pol)
    #polygon, _ = ox.project_geometry(polygon, crs={'init':'epsg:32633'}, to_latlong=True)
    g_zona = ox.graph_from_polygon(polygon, network_type='drive_service', retain_all = True)
    #intersezione fra nodi dentro la zona e quelli della rete della città
    intersezione = set(g.nodes()).intersection(set(g_zona.nodes()))
    for j in list(intersezione):
        #assegnazione del valore del quartiere al dizionario
        nodes_zona[j] = zone.loc[i,'Name'][18:]
        #assegnazione del valore dell'ID del quartiere al dizionario
        ID_nodes_zona[j] = i
        
print('Numero di nodi di cui è stata individuata la zona catastale:', len(nodes_zona))
print('Numero di nodi totali:', len(nodes))

#assegnazione del valore 'Quartiere non identificato' come valore di default
default_zona = {i:'zona non identificata' for i in set(g.nodes())}
default_ID_zona = {i:-1 for i in set(g.nodes())}
nx.set_node_attributes(g, default_zona, 'zona_catastale')
nx.set_node_attributes(g, default_ID_zona, 'ID_zona_catastale')

#assegnazione del quartiere corrispondente ad ogni nodo
nx.set_node_attributes(g, nodes_zona, 'zona_catastale')
nx.set_node_attributes(g, ID_nodes_zona, 'ID_zona_catastale')

nodes, edges = ox.graph_to_gdfs(g)
nodes.head()


# Rappresentazione della rete di Napoli evidenziando le zone catastali

# In[51]:


from random import random

colors = [(random(),random(),random(),1) for i in range(len(zone)+1)]
cc = [colors[i] for i in list(nodes.ID_zona_catastale)]

fig, ax = ox.plot_graph(g, node_color=cc, node_edgecolor=cc,
                        fig_height=15, fig_width=15,
                        show=False, close=False,
                        node_size=11, node_zorder=2, edge_linewidth=0.4,
                       margin=0)

#si disegnano sopra i confini del quartiere presi dal file .shp
for i in range(n_zone):
    pol = zone.loc[i,'WKT']
    polygon = shapely.wkt.loads(pol)
    polypatch = PolygonPatch(polygon, alpha=1, zorder=2, color = 'black', fill = False)
    ax.add_patch(polypatch)
    
fig_file_name = os.path.join(picture_dir, 'rete_napoli_zone_catastali')
plt.savefig(fig_file_name,bbox_inches='tight')

plt.show()


# Salvo la rete completa in un file

# In[34]:


objects_dir = os.path.join(current_dir, 'objects')
network_file_name = os.path.join(objects_dir, 'napoli.network')

with open(network_file_name, 'wb') as napoli_network_file:
    pickle.dump(g, napoli_network_file)

