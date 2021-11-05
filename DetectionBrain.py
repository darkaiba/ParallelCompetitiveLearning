#Importação das bibliotecas para construção e análise da rede
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from kmodes.kmodes import KModes
import pandas as pd
import statistics as st
from ParallelCompetitiveLearning.ParallelCompetitve import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
#from mayavi import mlab
import community as com
import itertools
import ast
from scipy.io import loadmat
#mlab.options.offscreen = True

def positionNodes(graph, lista):
    xy = {}
    yz = {}
    zx = {}

    max = 0
    min = 0
    for l in lista:
        for num in l:
            max = max if max > float(num) else float(num)
            min = min if min < float(num) else float(num)

    min *= (-1)

    no = list(nx.nodes(graph))
    number_nodes = 0

    for dado in lista:
        d = dado
        xy.update({no[number_nodes]: np.array([float(d[0])/(max if float(d[0]) > 0 else min), float(d[1])/(max if float(d[1]) > 0 else min)])})
        yz.update({no[number_nodes]: np.array([float(d[1])/(max if float(d[1]) > 0 else min), float(d[2])/(max if float(d[2]) > 0 else min)])})
        zx.update({no[number_nodes]: np.array([float(d[0])/(max if float(d[0]) > 0 else min), float(d[2])/(max if float(d[2]) > 0 else min)])})
        number_nodes += 1

    return xy, yz, zx

# Nesta primeira etapa é feita a leitura do arquivo contendo os vértices e as ligações, salva as conexões em um vetor
print('=============================================================================================================')
t = ['Ponderado', 'Binario']
rede = ['Normal']

for tipo in t:
    print('Rede: {}'.format(tipo))
    for r in rede:
        print('Grupo: {}'.format(r))
        vec_Con = []

        # Criação da Rede com seus vértices e suas respectivas ligações
        #Dados
        #Atlas AAL --->  versão 3.1 com 166 ROIs (Regiões de interesse)
        #Matriz média de 54 pacientes (pre-processados) com Alzheimer tirados do banco de dados da OASIS
        graph = nx.Graph()

        nodes = []
        NN = []
        arquiv = open('Arquivos/Nodes.csv','r')
        for linha in arquiv:
            nodes.append('{}'.format(linha.replace('\n','')))
        arquiv.close()

        for i in nodes:
            nodes = i.split(';')
            no = nodes[1].split(',')
            graph.add_node(nodes[0], pos=(float(no[0]), float(no[1]), float(no[2])))
            NN.append(nodes[0])

        # carregando o arquivo matlab
        filename = "Arquivos/Ma_{}.txt".format(r)
        arquiv = open(filename,'r')

        nnn = []
        for linha in arquiv:
            l = linha.replace('\n', '').split('\t')
            nn = []
            for i in l:
                nn.append(float(i))
            nnn.append(nn)
        arquiv.close()
    
        if tipo == 'Ponderado':
            for i in range(len(nnn)):
                for j in range(len(nnn[i])):
                    if nnn[i][j] > 0.2:
                        graph.add_edge(NN[i], NN[j], weight=nnn[i][j])
        else:
            for i in range(len(nnn)):
                for j in range(len(nnn[i])):
                    if nnn[i][j] > 0.2:
                        graph.add_edge(NN[i], NN[j])


        # Plotagem da Matriz de Conectividade da Rede
        print('Matriz')
        G = nx.from_numpy_matrix(nx.adjacency_matrix(graph).todense())
        B = nx.to_numpy_matrix(G)

        plt.matshow(B)
        plt.colorbar()
        plt.savefig('Matriz_{}.png'.format(tipo))

        # Fazendo a chamada do método de deteção de comunidades do algoritmo de Competição de Partículas
        print('Rede Construida')
        print('=============================================================================================================')
        arc_pos = nx.get_node_attributes(graph, 'pos')

        # Lista de cores para colorir as comunidades detectadas
        colors = ['red','blue','yellow','green','purple','pink','orange','darkred','black','cyan','magenta','lightyellow','lightblue','lightgreen','darkgray']

        # Posicionamento de Vértices na Rede
        posA, posB, posC = positionNodes(graph, arc_pos.values())
        edge_col = []
        arc_weight = nx.get_edge_attributes(graph, 'labels').values()
        for co in arc_weight:
            edge_col.append('Gray' if float(co) < 0 else 'Black')
        edge_col = 'Black'
        labels = [x+1 for x in range(graph.number_of_nodes())]
        dic_labels = {}
        for m, n in zip(labels, list(graph.nodes())):
            dic_labels[n] = m

        print('=========================================================================================================')
        print('\tFind Communities with ParallelCompetitive... ')
        #Detectando Módulos na Rede para 5 comunidades. Escolhemos 5 para ficar o mais próximo do Cortex Esquemático de Cérebro
        """
        A. Machado, Neuroanatomia funcional ser. Biblioteca biomedica. Atheneu, 1999, [online] Available: https://books.google.com.br/books?id=37NMPgAACAAJ.
        W. A. A. da Rocha, "Aspectos de redes complexas com aplicacoes em neurociencia", Dissertacão (Mestrado em Matemdtica Aplicada) - Universidade Estadual de Campinas Campinas (Sao Paulo Brasil), no. 76–91, 2017.
        """
        p = ParallelCompetitive()

        if r == 'Ponderado':
            parallelCompetitive, rk = p.interate(graph, 5, ponderado=True)
        else:
            parallelCompetitive, rk = p.interate(graph, 5, ponderado=False)

        # Preparando plotagem...
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 7))
        plt.subplot(131)
        plt.title('From Behind')
        img = plt.imread('Arquivos/brain1.png')
        plt.imshow(img, extent=[-0.9, 1.1, -0.8, 1.1])
        nx.draw_networkx_edges(graph, posC, edge_color='gray', width=1.0, alpha=0.4)
        num = 0
        for part in parallelCompetitive:
            nx.draw_networkx_nodes(graph, posC, graph.subgraph(part), node_color=colors[num], node_size=50, alpha=1.0)
            nx.draw_networkx_edges(graph.subgraph(part), posC, edge_color=colors[num], width=1.0, alpha=0.5)
            num += 1
            if num >= len(colors) or part == len(parallelCompetitive):
                num = 0

        plt.subplot(132)
        plt.title('Right Side')
        nx.draw_networkx_edges(graph, posB, edge_color='gray', width=1.0, alpha=0.4)
        img = plt.imread('Arquivos/brain2.png')
        plt.imshow(img, extent=[-1.2, 1.0, -1.1, 1.1])
        num = 0
        for part in parallelCompetitive:
            nx.draw_networkx_nodes(graph, posB, graph.subgraph(part), node_color=colors[num], node_size=50, alpha=1.0)
            nx.draw_networkx_edges(graph.subgraph(part), posB, edge_color=colors[num], width=1.0, alpha=0.5)
            num += 1
            if num >= len(colors) or part == len(parallelCompetitive):
                num = 0

        plt.subplot(133)
        plt.title('Upper')
        img = plt.imread('Arquivos/brain3.png')
        plt.imshow(img, extent=[-0.9, 1.1, -1.1, 0.9])
        nx.draw_networkx_edges(graph, posA, edge_color='gray', width=1.0, alpha=0.4)
        num = 0
        for part in parallelCompetitive:
            nx.draw_networkx_nodes(graph, posA, graph.subgraph(part), node_color=colors[num], node_size=50, alpha=1.0)
            nx.draw_networkx_edges(graph.subgraph(part), posA, edge_color=colors[num], width=1.0, alpha=0.5)
            num += 1
            if num >= len(colors) or part == len(parallelCompetitive):
                num = 0

        plt.savefig('ParallelCompetitive_{}.png'.format(tipo))

        #Avaliando as Qualidades Modulares Proposta por Fortunato
        """
        Santo Fortunato. “Community Detection in Graphs”. Physical Reports, Volume 486, Issue 3–5 pp. 75–174 <https://arxiv.org/abs/0906.0612>
        """
        print('\tModules Quality')
        vetorTecnicas = ['Proposed Technique']
        vetorMod = [nx.algorithms.community.quality.modularity(graph, parallelCompetitive)]
        vetorPer = [nx.algorithms.community.quality.performance(graph, parallelCompetitive)]
        vetorCov = [nx.algorithms.community.quality.coverage(graph, parallelCompetitive)]

        x1 = np.arange(len(vetorMod))
        x2 = [x + 0.25 for x in x1]
        x3 = [x + 0.25 for x in x2]

        plt.figure(figsize=(8, 8))
        plt.bar(x1, vetorMod, label='Modularity', width=0.25, color = 'r')
        plt.bar(x2, vetorPer, label='Performance', width=0.25, color = 'b')
        plt.bar(x3, vetorCov, label='Coverage', width=0.25, color = 'g')
        coisa = ['Parallel Competition']
        plt.xticks([x + 0.25 for x in range(len(coisa))], coisa)
        plt.ylabel("Value")
        plt.legend()
        plt.savefig('Quality_{}.png'.format(tipo))

    # Final
    print('================================================================================================================')
    print('Processo de Detecção e Análise de Comunidades em Redes Cerebrais Finalizada')
    print('================================================================================================================')
