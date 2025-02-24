#Importação das bibliotecas para construção e análise da rede
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import igraph as ig
from ParallelCompetitive import *
import random
import community as com
from scipy.cluster import hierarchy
import plotly.figure_factory as ff
import matplotlib.cm as cm
import itertools

#Criação da Rede GN
def girvan_graphs(zout):
    pout = float(zout) / 96.
    pin = (16. - pout * 96.) / 31.
    graph = nx.Graph()
    graph.add_nodes_from(range(128))
    for x in graph.nodes():
        for y in graph.nodes():
            if x != y:
                val = random.random()
                if x % 4 == y % 4:
                    if val < pin:
                        graph.add_edge(x, y)
                        graph.add_edge(y, x)

                else:
                    if val < pout:
                        graph.add_edge(x, y)
                        graph.add_edge(y, x)
    return graph

#Iniciando o Construtor
print('Iniciando o Algoritmo de Aprendizado Competitivo Paralelo')
pc = ParallelCompetitive()

print('Criando Redes e Detectando Comunidades')
#Iniciando a detecção de comunidades em Alguns Exemplos
# Benchmark de GIRVAN E NEWMAN
print('\t GN')
graphGN = girvan_graphs(0.2*16)#4comunidades
compParticleGN, R_K_GN = pc.interate(graphGN, 4, checkEnergy=True)

#Benchmark Lancichietti(LFR)
print('\t LFR')
graphLFR = nx.generators.community.LFR_benchmark_graph(500, 3, 1.5, 0.05, average_degree=20, min_community=120, seed=40)#3comunidades
compParticleLFR, R_K_LFR = pc.interate(graphLFR, 3, checkEnergy=True)

#Benchmark do Clube de KARATE
print('\t Karate')
graphKC = nx.karate_club_graph()#2comunidades
compParticleKC, R_K_KC = pc.interate(graphKC, 2, checkEnergy=True)

#Desbalanceamento em Densidade
print('\t Desbalanceamento Densidade')
graphUD = nx.generators.community.random_partition_graph([100, 100], 0.1, 0.01)#2comunidades
compParticleUD, R_K_UD = pc.interate(graphUD, 2, checkEnergy=True)

#Desbalanceamento em Tamanho
print('\t Desbalanceamento Tamanho')
graphUS = nx.generators.community.random_partition_graph([200, 50, 100, 10], 1.0, 0.01)#4comunidades
compParticleUS, R_K_US = pc.interate(graphUS, 4, checkEnergy=True)

#Vetor de Cores
colors = ['red','blue','yellow','green','purple','pink','orange','darkred','black','darkgray','cyan','magenta','lightyellow','lightblue','lightgreen']

print('Plotando....')
# Plotagem da rede em grafos
fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(12, 6))
fig.suptitle('Parallel Competition Learning')
#Plotagem Rede GN
plt.subplot(231)
posA = nx.kamada_kawai_layout(graphGN)
num = 0
labels = [x+1 for x in range(graphGN.number_of_nodes())]
dic_labels = {}
for m, n in zip(labels, list(graphGN.nodes())):
    dic_labels[n] = m
plt.title('Girvan & Newman')
for j in compParticleGN:
    nx.draw_networkx_nodes(graphGN, posA, graphGN.subgraph(j), node_color=colors[num], node_size=50, alpha=1.0)
    nx.draw_networkx_edges(graphGN, posA, edge_color='black', width=1.0, alpha=0.2)
    num += 1

#Plotagem Rede LFR
plt.subplot(232)
posA = nx.kamada_kawai_layout(graphLFR)
num = 0
labels = [x+1 for x in range(graphLFR.number_of_nodes())]
dic_labels = {}
for m, n in zip(labels, list(graphLFR.nodes())):
    dic_labels[n] = m
plt.title('Lancichietti')
for j in compParticleLFR:
    nx.draw_networkx_nodes(graphLFR, posA, graphLFR.subgraph(j), node_color=colors[num], node_size=50, alpha=1.0)
    nx.draw_networkx_edges(graphLFR, posA, edge_color='black', width=1.0, alpha=0.2)
    num += 1

#Plotagem Rede do Clube de Karate
plt.subplot(233)
posA = nx.kamada_kawai_layout(graphKC)
num = 0
labels = [x+1 for x in range(graphKC.number_of_nodes())]
dic_labels = {}
for m, n in zip(labels, list(graphKC.nodes())):
    dic_labels[n] = m
plt.title('Karate CLube')
for j in compParticleKC:
    nx.draw_networkx_nodes(graphKC, posA, graphKC.subgraph(j), node_color=colors[num], node_size=50, alpha=1.0)
    nx.draw_networkx_edges(graphKC, posA, edge_color='black', width=1.0, alpha=0.2)
    num += 1

#Plotagem Rede Desbalanceada Densidade - 1
plt.subplot(234)
posA = nx.kamada_kawai_layout(graphUD)
num = 0
labels = [x+1 for x in range(graphUD.number_of_nodes())]
dic_labels = {}
for m, n in zip(labels, list(graphUD.nodes())):
    dic_labels[n] = m
plt.title('Desbalanceamento Densidade - 1')
for j in compParticleUD:
    nx.draw_networkx_nodes(graphUD, posA, graphUD.subgraph(j), node_color=colors[num], node_size=50, alpha=1.0)
    nx.draw_networkx_edges(graphUD, posA, edge_color='black', width=1.0, alpha=0.2)
    num += 1

#Plotagem Rede Desbalanceada Densidade - 2
plt.subplot(235)
posA = nx.kamada_kawai_layout(graphUD)
num = 0
labels = [x+1 for x in range(graphUD.number_of_nodes())]
dic_labels = {}
for m, n in zip(labels, list(graphUD.nodes())):
    dic_labels[n] = m
plt.title('Desbalanceamento Densidade - 2')
for j in compParticleUD:
    nx.draw_networkx_nodes(graphUD, posA, graphUD.subgraph(j), node_color=colors[num], node_size=50, alpha=1.0)
    nx.draw_networkx_edges(graphUD, posA, edge_color='black', width=1.0, alpha=0.2)
    num += 1

#Plotagem Rede Desbalanceada Tamanho
plt.subplot(236)
posA = nx.kamada_kawai_layout(graphUS)
num = 0
labels = [x+1 for x in range(graphUS.number_of_nodes())]
dic_labels = {}
for m, n in zip(labels, list(graphUS.nodes())):
    dic_labels[n] = m
plt.title('Desbalanceamento Tamanho')
for j in compParticleUS:
    nx.draw_networkx_nodes(graphUS, posA, graphUS.subgraph(j), node_color=colors[num], node_size=50, alpha=1.0)
    nx.draw_networkx_edges(graphUS, posA, edge_color='black', width=1.0, alpha=0.2)
    num += 1

plt.show()
# Final
print('================================================================================================================')
print('Processo Finalizado')
print('================================================================================================================')

