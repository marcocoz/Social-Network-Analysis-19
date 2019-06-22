#!/usr/bin/env python
# coding: utf-8

# # Open Question

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
import osmnx as ox
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from statistics import mean
import geopandas as gpd
from shapely.wkt import loads
from shapely.ops import unary_union
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from shapely.geometry import Point
from shapely.ops import cascaded_union
from rtree import index
from descartes.patch import PolygonPatch
import seaborn as sns

current_dir = os.getcwd()
picture_dir = os.path.join(current_dir, 'grafici')
objects_dir = os.path.join(current_dir, 'objects')
network_file_name = os.path.join(objects_dir, 'napoli.network')

with open(network_file_name, 'rb') as napoli_network_file:
    g = pickle.load(napoli_network_file)
    
nodes, edges = ox.graph_to_gdfs(g)
nodes.head()


# In[15]:


get_ipython().system('pip install statsmodel')


# Si definiscono dei flag che indicano se un nodo è un particolare punto di interesse

# In[2]:


nodes['flag_castello'] = 0
nodes['flag_ospedale'] = 0
nodes['flag_parco'] = 0
nodes['flag_scuola'] = 0
nodes['flag_universita'] = 0

nodes.loc[nodes['ID_castle'] != -1, 'flag_castello'] = 1
nodes.loc[(nodes['ID_ospedale'] != -1)&(nodes['ID_ospedale_comune'] != -1), 'flag_ospedale'] = 1
nodes.loc[nodes['ID_parco'] != -1, 'flag_parco'] = 1
nodes.loc[nodes['ID_scuola'] != -1, 'flag_scuola'] = 1
nodes.loc[nodes['ID_universita'] != -1, 'flag_universita'] = 1

nodes['flag_nodo_interesse'] = nodes['flag_castello']+nodes['flag_ospedale']+nodes['flag_parco']+nodes['flag_scuola']+nodes['flag_universita']


# Conteggio vari punti di interesse per quartiere

# In[7]:


columns = ['zona_catastale', 'flag_castello', 'flag_ospedale', 'flag_parco', 'flag_scuola','flag_universita', 'flag_nodo_interesse']
zone_catastali = nodes[columns].groupby('zona_catastale').sum()
zone_catastali['zona_catastale'] = zone_catastali.index
zone_catastali = zone_catastali[zone_catastali['zona_catastale'] != 'zona non identificata']
zone_catastali.index = range(len(zone_catastali))
zone_catastali = zone_catastali.rename(columns={'zona_catastale':'Zona'})
zone_catastali.head()


# ### Statistiche zone OMI

# Calcolo delle statistiche della rete stradale di ogni quartiere.
# 
# ATTENZIONE: richiede qualche minuto! Per questo motivo la tabella è stata salvata nel file statistiche_quartieri.csv
# 
# CONSIGLIO: passare alla cella successiva che carica i risultati da file csv

# In[211]:


file_name = os.path.join(current_dir, 'napoli_omi.csv')
zone = gpd.read_file(file_name)
n_zone = len(zone)
nodes_quartier = dict()
ID_nodes_quartier = dict()
keys = ['n', 'm', 'k_avg', 'streets_per_node_avg', 'edge_length_total', 'edge_length_avg', 'street_length_total', 'street_length_avg', 'circuity_avg']
records = list()

for i in range(n_zone):
    #estrazione del poligono relativo alla zona catastale
    pol = zone.loc[i,'WKT']
    polygon = loads(pol)
    g_zona = ox.graph_from_polygon(polygon, network_type='drive_service', retain_all = True)
    polygon, _ = ox.project_geometry(polygon, to_latlong=False)
    area = unary_union(polygon).area/10**6
    try:
        stats = ox.basic_stats(g_zona)
        stats = dict((k,stats[k]) for k in keys)
        stats['area'] = area
        stats['Zona'] = zone.loc[i,'Name'][18:]
        stats['avg_betweenness'] = mean(nx.betweenness_centrality(g_zona).values())
        stats['avg_closeness'] = mean(nx.closeness_centrality(g_zona).values())
        G = nx.DiGraph()
        for u,v,data in g_zona.edges(data=True):
            w = data['length']# if 'length' in data else 1.0
            if not G.has_edge(u,v):
                G.add_edge(u, v, weight=w)
        stats['avg_clustering'] = mean(nx.clustering(G).values())
    except:
        stats = dict()
        stats['area'] = area
        stats['Zona'] = zone.loc[i,'Name'][18:]
    records.append(stats)
    
stats_zona = pd.DataFrame(records, columns=keys+['area','Zona','avg_betweenness','avg_closeness','avg_clustering'])
stats_zona.to_csv('statistiche_zone_cat.csv', sep=';', index=False)


# In[8]:


#caricamento dati geografici delle zone OMI
file_name = os.path.join(current_dir, 'napoli_omi.csv')
zone = gpd.read_file(file_name)
n_zone = len(zone)

#caricamento statistiche zone OMI
stats_zona = pd.read_csv('statistiche_zone_cat.csv', sep=';')
stats_zona.head()


# Si indaga per vedere perchè la zona R1 non ha individuato alcun network. Come si vede dall'immagine sotto è presente una rete se consideriamo anche le strade pedonali mentre la rete sparisce nel momento in cui consideariamo solo le strade percorribili in automobile

# In[9]:


#si estrae il network che comprende anche le strade pedonali
g = ox.graph_from_place('Napoli, Italia', network_type="walk") #network_type="drive_service"
fig, ax = ox.plot_graph(g,# node_color=cc, node_edgecolor=cc,
                        fig_height=10, fig_width=10,
                        show=False, close=False,
                        node_size=11, node_zorder=2, edge_linewidth=0.4,
                       margin=0)

#si disegnano sopra i confini del quartiere presi dal file .shp
#for i in range(n_zone):
pol = zone.loc[1,'WKT']
polygon = loads(pol)
polypatch = PolygonPatch(polygon, alpha=1, zorder=2, color = 'black', fill = False)
ax.add_patch(polypatch)


# ### Stima numero di abitanti zone catastali

# Calcolo dell'area comune tra tutte le coppie (zona catastale, quartiere)

# In[4]:


quartieri_directory = os.path.join(current_dir, 'Quartieri')
quartieri_napoli = gpd.read_file(quartieri_directory)

records = list()

for i in range(len(zone)):
    for j in range(len(quartieri_napoli)):
        p_zona = loads(zone.loc[i,'WKT'])
        p_zona, _ = ox.project_geometry(p_zona, to_latlong=False)
        p_quart = quartieri_napoli['geometry'].iloc[j]
        polygons = [p_quart,p_zona]

        intersections = []
        idx = index.Index()

        for pos, polygon in enumerate(polygons):
            idx.insert(pos, polygon.bounds)

        for polygon in polygons:
            merged_polygons = cascaded_union([polygons[pos] for pos in idx.intersection(polygon.bounds) if polygons[pos] != polygon])
            intersections.append(polygon.intersection(merged_polygons))

        intersection = cascaded_union(intersections)
        if intersection.area > 0:
            record = dict()
            record['Zona'] = zone.loc[i,'Name'][18:]
            record['Quartiere'] = quartieri_napoli.loc[j,'QUART']
            record['area_zona'] = p_zona.area/10**6
            record['area_quartiere'] = p_quart.area/10**6
            record['area_comune'] = intersection.area/10**6
            records.append(record)

df_area = pd.DataFrame(records,columns=['Zona','Quartiere','area_zona','area_quartiere','area_comune'])
print(df_area['area_comune'].sum())
df_area.head()


# Si caricano i dati contenenti il numero di abitanti per quartiere. Si utilizza questo dato per stimare la popolazione delle zone OMI. Per fare questo si assume che la popolazione sia equamente distribuita all'interno dei quartieri

# In[11]:


abit_quart = pd.read_csv('abitanti_quartieri_2011.csv')
df_area_ab = pd.merge(df_area,abit_quart)
df_area_ab['n_abitanti'] = round(df_area_ab['area_comune']/df_area_ab['area_quartiere']*df_area_ab['n_abitanti_quart'])
df_abit = df_area_ab[['Zona','n_abitanti']].groupby('Zona',as_index=False).sum()


# ### Unione dei dati

# Si uniscono le informazioni per l'analisi e si creano le variabili di interesse

# In[12]:


omi_dir = os.path.join(current_dir, 'QUO_NA')
file_name = os.path.join(omi_dir, 'QI_681_1_20152_VALORI.csv')
df_omi = pd.read_csv(file_name, sep=';',skiprows=1)

df = pd.merge(zone_catastali,stats_zona,on='Zona')
df = pd.merge(df,df_omi,on='Zona')
col_to_rename = {'flag_castello':'n_castelli', 'flag_ospedale':'n_ospedali', 'flag_parco':'n_parchi',
                 'flag_scuola':'n_scuole', 'flag_universita':'n_universita',#'zona':'Zona',
                 'flag_nodo_interesse':'n_nodi_interesse', 'n':'n_nodes','m':'n_edges'}

df = df.rename(columns=col_to_rename)

print('Number of rows: {}\nNumber of columns: {}'.format(len(df),len(df.columns)))
print('Number of rows omi:',len(df_omi))

df = pd.merge(df,df_abit, on='Zona')
df['ratio_nodi_interesse'] = df['n_nodi_interesse']/df['n_abitanti']
df['ratio_edges'] = df['n_edges']/df['n_abitanti']
df['prezzo_medio'] = (df['Compr_min']+df['Compr_max'])/2
df.head()


# ### Correlazione

# Si indaga sulla correlazione tra le variabili di interesse

# ##### Pearson Correlation Coefficient

# In[13]:


from statsmodels.graphics.correlation import plot_corr

columns = ['k_avg','street_length_avg','ratio_nodi_interesse','ratio_edges',
          'avg_betweenness','avg_closeness','avg_clustering','prezzo_medio']

fig = plot_corr(df[columns].corr(), xnames=columns)
fig.set_size_inches(12, 9)


# In[16]:


df[columns].corr()


# ##### Kendall Rank Correlation Coefficient

# In[14]:


from scipy.stats import kendalltau

kendalltau_matrix = df[columns].corr()
for col1 in columns:
    for col2 in columns:
        kendalltau_matrix.loc[col1,col2] = kendalltau(np.array(df[col1]),np.array(df[col2]))[0]
        
fig = plot_corr(kendalltau_matrix, xnames=columns)
fig.set_size_inches(12, 9)
fig_file_name=os.path.join(picture_dir)
plt.savefig(fig_file_name,bbox_inches='tight')


# In[23]:


kendalltau_matrix


# ### Indagine manuale

# Si effettua un'indagine manuale attraverso l'utilizzo di opportuni threshold per indagare sulle relazioni tra rete stradale, numero di nodi di interesse e prezzo medio

# In[24]:


threshold = 0.00005
df['flag_zona_interessante'] = df['ratio_nodi_interesse'] < threshold
#introduco il threshold: i comuni con il threshold minore di una certa soglia (true) necessitano di una riqualificazione
df_int = df.loc[df['Cod_Tip']==20,['Zona', 'ratio_nodi_interesse', 'flag_zona_interessante','prezzo_medio']]

df['prezzo_medio'] = (df['Compr_min']+df['Compr_max'])/2
zone_riq = df_int.loc[df['flag_zona_interessante']==True,'Zona']
df_int[df_int['flag_zona_interessante']==True]


# Definizione tabella relativa alla rete e di un opportuno threshold

# In[25]:


threshold = 0.02
df['flag_rete_stradale'] = df['ratio_edges'] < threshold
#introduco il threshold: i comuni con il threshold minore di una certa soglia (true) necessitano di una riqualificazione
df_rete = df.loc[df['Cod_Tip']==20,['Zona', 'ratio_edges', 'flag_rete_stradale','prezzo_medio']]

zone_riq_rete = df_rete.loc[df['flag_rete_stradale']==True,'Zona']
df_rete[df_rete['flag_rete_stradale']==True]


# Unione delle due tabelle per un'analisi incrociata

# In[26]:


pd.merge(df_rete,df_int)


# ### Analisi temporale

# ##### Periodo 2002-2013

# Si caricano i dati relativi alle quotazioni immobiliari dal 2002 al 2013 e si riuniscono in un unico dataset

# In[27]:


omi_dir = os.path.join(current_dir, 'QUO_NA')
n_periods = 23
file_name = os.path.join(omi_dir, 'QI_681_5_20132_VALORI.csv')
df_omi = pd.read_csv(file_name, sep=';',skiprows=1)
df_omi['semester'] = 20132

for i in range(1,n_periods):
    file_name = os.path.join(omi_dir, 'QI_681_'+str(i+5)+'_'+str(2013-i//2)+str((i+1)%2+1)+'_VALORI.csv')
    df_tmp = pd.read_csv(file_name, sep=';',skiprows=1)
    df_tmp['semester'] = int(str(2013-i//2)+str((i+1)%2+1))
    df_omi = pd.concat([df_omi,df_tmp])


# Si raggruppano poi i dati per zona catastale

# In[28]:


trend_model = LinearRegression(normalize=True, fit_intercept=True)
df_omi['prezzo_medio'] = (df_omi['Compr_min']+df_omi['Compr_max'])/2
#si elimina la zona E9 perchè contiene pochissimi valori
df_omi = df_omi[df_omi['Zona']!='E9']
cols = ['Zona','prezzo_medio','semester']
price_sems = df_omi.loc[df_omi['Cod_Tip'] == 20, cols].groupby(['Zona'],as_index=False)
ts = [x[1] for x in price_sems]
for x in ts:
    x.set_index('semester',inplace=True)
ts = [(x['Zona'].iloc[0],x['prezzo_medio'], trend_model.fit(np.array(x['prezzo_medio'].index).reshape((-1,1)), x['prezzo_medio']).coef_[0]) for x in ts]


# Si rappresentano le time series. In ogni grafico sono presenti le time series di tutte le zone catastali

# In[29]:


plt.figure(figsize=(10,8))

for ts_zona in ts:
    if ts_zona[0] not in ['E8','E17']:
        plt.plot(ts_zona[1], color='lime')
    elif ts_zona[0] == 'E8':
        plt.plot(ts_zona[1], color='r')
    else:
        plt.plot(ts_zona[1], color='b')
    
legend_elements = [Line2D([0], [0], marker='_', color='r', label='E8', linestyle=''),
                   Line2D([0], [0], marker='_', color='b', label='E17', linestyle='')]
    
plt.title('Time Series Zone')
plt.legend(handles=legend_elements)
fig_file_name = os.path.join(picture_dir, 'ts')
plt.savefig(fig_file_name,bbox_inches='tight')

plt.show()


# Si crea una tabella contenente il trend relativo ad ogni time series, si estrae la media e si rappresenta la distribuzione del trend all'interno delle zone

# In[30]:


records = [(x[0], x[2]) for x in ts]
df_ts = pd.DataFrame(records, columns=['Zona', 'Trend'])
sns.distplot(df_ts['Trend'])
fig_file_name = os.path.join(picture_dir, 'ts_trend')
plt.savefig(fig_file_name,bbox_inches='tight')
print('Trend medio:',df_ts['Trend'].mean())
print('Rank zona E8:',len(df_ts[df_ts['Trend']>df_ts.loc[(df_ts['Zona']=='E8'), 'Trend'].values[0]])+1)
print('Rank zona E17:',len(df_ts[df_ts['Trend']>df_ts.loc[(df_ts['Zona']=='E17'), 'Trend'].values[0]])+1)
print('Numero totale di zone:',len(df_ts))
df_ts[(df_ts['Zona']=='E8') | (df_ts['Zona']=='E17')]

