#Importação da biblioteca NetworkX para construção e análise da rede
import networkx as nx
import numpy as np
import math
import random

class ParallelCompetitive():
    #construtor inicial
    def __init__(self):
        self.Energy = []
        self.k = []
        self.signal = 1
        self.alpha = 0.4

    #Função para iniciar a energia sináptica em zero
    def initialEnergy(self, graph, numParticle):
        E = []
        nodos = list(nx.nodes(graph))

        for i in range(0, numParticle):
            vet = []
            for j in range(0, len(nodos)):
                vet.append(0)
            E.append(vet)

        return E

    #Função para iniciar o peso de regularização em um
    def initialWeighted(self, graph, numParticle):
        P = []
        nodos = list(nx.nodes(graph))

        for i in range(0, numParticle):
            vet = []
            for j in range(0, len(nodos)):
                vet.append(1)
            P.append(vet)

        return P

    #Função para iniciar aleatóriamente as particulas
    def initialParticle(self, graph, numParticle):
        k = []
        nodos = list(nx.nodes(graph))

        dic = list(nx.closeness_centrality(graph).values())

        for cont in range(0, numParticle):
            max = -1
            pos = -1
            for value in range(0, len(dic)):
                if dic[value] > max:
                    max = dic[value]
                    pos = value

            k.append(nodos[pos])
            dic[pos] = -1
        return k

    #Função para propagar o sinal sinático e calcular a energia das particulas no vértices
    def propagationSignal(self, listParticle, graph, dic, ponderado, peso):
        nodos = list(nx.nodes(graph))

        for particle in range(0, len(listParticle)):
            j = nodos.index(self.k[particle])

            #aumenta a energia da particula visitada
            self.Energy[particle][j] += self.signal

            #dimuniu a energia das particulas rivais
            for v in range(0, len(self.k)):
                if v != particle:
                    self.Energy[v][j] -= self.signal

            #Propaga o sinal para os vizinhos
            vizinhos = list(nx.neighbors(graph, nodos[j]))

            for neighbors in range(0, len(vizinhos)):
                li = nodos.index(vizinhos[neighbors])

                #aumenta a energia dos vizinhos
                pri = nodos[j]
                sec = vizinhos[neighbors]
                if (pri, sec) not in dic:
                    pri = vizinhos[neighbors]
                    sec = nodos[j]

                self.Energy[particle][li] += self.signal * (math.fabs(float(dic[pri, sec])) if ponderado is True else 1)

                #dimuniu a energia dos vizinhos para particulas rivais
                for v in range(0, len(self.k)):
                    if v != particle:
                        self.Energy[v][li] -= self.signal * (math.fabs(float(dic[pri, sec])) if ponderado is True else 1)

    #check se a energia esta zerada ou passou de 100%
    def checkEnergy(self, graph, k):
        nodes = list(nx.nodes(graph))
        for i in range(0, len(nodes)):
            if self.Energy[k][i] < 0:
                self.Energy[k][i] = 0

    #Calculo da Matriz de Probabilidade para os nós não visitados
    def calculateM(self, g, numParticle, p):
        M = []
        nodos = list(nx.nodes(g))
        for linha in range(numParticle):
            vet = []
            Free = False
            for coluna in range(len(nodos)):
                if self.Energy[linha][coluna] >= 0:
                    Free = True
                    for i in range(numParticle):
                        if i != p and self.Energy[i][coluna] > 0:
                            Free = False

                if Free:
                    vet.append(1.0)
                else:
                    vet.append(0.0)

            M.append(vet)

        return M

    #A partícula escolhe um novo nó para visitar com base na probabilidade de duas matrizes
    def choiceNodos2(self, M, E, g, pos, peso, n):
        nodes = list(nx.nodes(g))
        somaM = 0
        somaE = 0
        vM = []
        vE = []

        for coluna in range(0, len(M[pos])):
            if M[pos][coluna] > 0:
                somaM += M[pos][coluna]#*peso[pos][coluna]

            if E[pos][coluna]*peso[pos][coluna] > 0:
                somaE += E[pos][coluna]*peso[pos][coluna]

        for coluna in range(0, len(M[pos])):
            if M[pos][coluna] <= 0:
                vM.append(0.0)
            else:
                vM.append((M[pos][coluna])/(somaM))

            if E[pos][coluna]*peso[pos][coluna] <= 0:
                vE.append(0.0)
            else:
                vE.append((E[pos][coluna]*peso[pos][coluna])/(somaE))

        Prean = []
        for linha in range(n):
            Prean.append([])
            for coluna in range(len(nodes)):
                Prean[linha].append(0.0)

        for linha in range(n):
            vet = []
            for coluna in range(len(nodes)):
                if self.Energy[linha][coluna] > 0:
                    Prean[linha][coluna] = 1.0
                    for i in range(n):
                        if i != pos and self.Energy[i][coluna] > 0:
                            Prean[linha][coluna] = 0.0

        
        vectorProb = (np.array(vM)*self.alpha) + (np.array(vE)*(1-self.alpha))

        if sum(vectorProb) < 1:
            dif = 1 - sum(vectorProb)
            name = nodes[np.argmax(vectorProb)]
            posM = nodes.index(name)
            vectorProb[posM] += dif
        elif sum(vectorProb) > 1:
            dif = sum(vectorProb) - 1
            name = nodes[np.argmax(vectorProb)]
            posM = nodes.index(name)
            vectorProb[posM] -= dif
        
        return np.random.choice(nodes, p=vectorProb)

    #Função para passar o resultado final das particulas
    def result(self, Energy, numParticle, graph):
        nodos = list(nx.nodes(graph))

        vk = []
        for tam in range(0, numParticle):
            vk.append([])

        posMaior = np.argmax(self.Energy, axis=0)

        for a in range(len(posMaior)):
            vk[posMaior[a]].append(nodos[a])

        return vk

    def R_K(self, A, g):
        norm1 = A/np.linalg.norm(A, axis=0)
        #print('A: {}'.format(A))
        #print('Normalizada: {}'.format(norm1))
        where_are_NaNs = np.isnan(norm1)
        norm1[where_are_NaNs] = 0
        return np.sum(np.max(norm1, axis=0))/g.number_of_nodes()

    def has_converged(self, A) -> bool:
        return A < 0.05
   
    #Função de iteração principal
    def interate(self, graph, numParticle, ponderado=False, checkEnergy=True):
        self.Energy = self.initialEnergy(graph, numParticle)#Inicia as energias sináptica zeradas
        self.k = self.initialParticle(graph, numParticle)#Esoclhe um nó aleatório para iniciar a partícula
        peso = self.initialWeighted(graph, numParticle)#Inicia os peso de regularização em um
        dic = nx.get_edge_attributes(graph, 'weights')#Cria o dicionário de links em caso de redes ponderadas

        interation = 0
        while interation < 1000:
            self.propagationSignal(self.k, graph, dic, ponderado, peso)#Propaga o sinal para o no e seus vizinhos
            for k in range(0, numParticle):
                if checkEnergy:
                    self.checkEnergy(graph, k)#Faz a checagem da energia

                M = self.calculateM(graph, numParticle, k)#Calcula a Matriz de propabilidade para os vértices não visitados
                self.k[k] = self.choiceNodos2(M, self.Energy, graph, k, peso, numParticle)#Escolhe um novo vértice para visitar
            interation += 1
        return self.result(self.Energy, numParticle, graph), self.R_K(self.Energy, graph)#Retorna o resultado das comunidades e o calculo R(t)
