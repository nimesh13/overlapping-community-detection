from collections import defaultdict
import argparse
import community as community_louvain
import networkx as nx
import numpy as py
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def project(filename, delimiter=' ', nodeType=int):
    graph = defaultdict(list)
    for line in open(filename, 'U'):
        L = line.strip().split(delimiter)
        graph[L[0]].append(L[1])
        graph[L[1]].append(L[0])

    return sorted(graph.items())

def createAdjacencyMatrix(graph):

    matrix = []
    num_nodes = len(graph)
    for vertices in graph:
        list = [0] * num_nodes
        for vertex in vertices[1]:
            vert = int(vertex) - 1
            list[vert] = 1

        matrix.append(list)

    return matrix

def createEdgeIdMatrix(matrix):
    counter = 0
    numNodes = len(matrix)
    edgeIdMatrix = [[-1]*numNodes for i in range(numNodes)]

    for i in range(0, numNodes):
        for j in range(i, numNodes):
            if matrix[i][j] == 1 :
                edgeIdMatrix[i][j] = counter
                edgeIdMatrix[j][i] = counter
                counter = counter + 1
    return edgeIdMatrix

def createLineGraph(graph, matrix, edgeIdMatrix):

    num_nodes = len(matrix)
    num_lines = 0
    for i in range(0, num_nodes):
        for j in range(0, i):
            if matrix[i][j] == 1:
                num_lines += 1

    line_graph = [[0] * num_lines for i in range(num_lines)]

    for i in range(0, num_nodes):
        nodeHub = i + 1
        adjCombList = getAdjCombinationOf2(graph, nodeHub)

        for comb in range(0, len(adjCombList)):
            nodeA = adjCombList[comb][0]
            nodeB = adjCombList[comb][1]

            nodeA_and_neighbors, nodeB_and_neighbors = getNodes(graph, nodeA, nodeB)
            nodeA_and_neighbors.append(nodeA)
            nodeA_and_neighbors.append(nodeHub)
            nodeB_and_neighbors.append(nodeB)
            nodeB_and_neighbors.append(nodeHub)

            similarity = findSimilarity(nodeA_and_neighbors, nodeB_and_neighbors)
            nodeA_id = edgeIdMatrix[nodeHub-1][nodeA-1]
            nodeB_id = edgeIdMatrix[nodeHub-1][nodeB-1]

            line_graph[nodeA_id][nodeB_id] = similarity
            line_graph[nodeB_id][nodeA_id] = similarity

    return line_graph

def findSimilarity(nodeA, nodeB):
    a = set(nodeA)
    b = set(nodeB)
    S = 1.0 * len(a & b) / len(a | b)
    return S

def getNodes(graph, nodeA, nodeB):
    valA = [item[1] for item in graph if item[0] == str(nodeA)]
    valB = [item[1] for item in graph if item[0] == str(nodeB)]
    adjA = valA[0][:]
    adjB = valB[0][:]
    return adjA, adjB

def getAdjCombinationOf2(graph, hubNode):
    # Find the edge that shared the same node to compare.
    # For example, edge u->v and edge s->v shared the same node, then v is acted as the hub node
    # This function wants to find all combinations of 2 nodes (u and s) that are adjacent to hub node (v)
    # to represents the combinations of two edges.
    result = []

    for item in graph:
        if item[0] == str(hubNode):
            adjNum = len(item[1])
            for i in range(0, adjNum-1):
                for j in range(i+1, adjNum):
                    comb = []
                    comb.append(int(item[1][i]))
                    comb.append(int(item[1][j]))
                    result.append(comb)
            break
    return result

def getCommunityById(edgeIdMatrix , allCommunityDict, communityId):
    lineNode = []

    for key, val in allCommunityDict.items():
        if int(val) == int(communityId):
            lineNode.append(key)

    numNodes = len(edgeIdMatrix)
    communityMatrix = [[0]*numNodes for i in range(numNodes)]

    for i in range(0, numNodes):
        for j in range(i, numNodes):
            for lineId in lineNode:
                if edgeIdMatrix[i][j] == lineId:
                    communityMatrix[i][j] = 1
                    communityMatrix[j][i] = 1

    return communityMatrix



parser = argparse.ArgumentParser()
parser.add_argument(dest='filename', help="Filename of the dataset containing the graph")

# Parse and print the results
filename = parser.parse_args().filename
# print(filename)

number_of_vertices = 0

graph = project(filename)
print('\nThe nodes with their neighbours are: \n')
for node in graph:
    print("%s -> %s" % (node[0], node[1]))

matrix = createAdjacencyMatrix(graph)
print('\nThe corresponding adjacency matrix is: \n')
for entry in range(0, len(matrix)):
    print(matrix[entry])

edgeIdMx = createEdgeIdMatrix(matrix)
print('\nThe edge ID adjacency matrix is: \n')
for entry in range(0, len(edgeIdMx)):
    print(edgeIdMx[entry])

line_graph = createLineGraph(graph, matrix, edgeIdMx)
print('\nThe line graph adjacency matrix with weights as similarity is: \n')
for entry in range(0, len(line_graph)):
    print(line_graph[entry])

# print(line_graph)
G = nx.Graph(py.matrix(line_graph))
print('************')
# print(G)

K = community_louvain.best_partition(G, weight='weight')
modularity2 = community_louvain .modularity(K, G, weight='weight')
print("The modularity Q based on networkx is {}".format(modularity2))

print('---------------')
print(K)

print()
print("community 0 matrx")
community0 = getCommunityById(edgeIdMx, K, 0)
for row in community0:
    print(row)

print("community 1 matrx")
community1 = getCommunityById(edgeIdMx, K, 1)
for row in community1:
    print(row)

print("community 2 matrx")
community2 = getCommunityById(edgeIdMx, K, 2)
for row in community2:
    print(row)


pos = nx.spring_layout(K)
cmap = cm.get_cmap('viridis', max(K.values()) + 1)
nx.draw_networkx_nodes(G, pos, K.keys(), node_size=800,
                       cmap=cmap, node_color=list(K.values()), label = True)
nx.draw_networkx_edges(G, pos, alpha=0.5)
# print('Labels: ', list(K.keys()))
# labels = 
nx.draw_networkx_labels(G, pos, K, font_size=22, font_color="whitesmoke")
plt.show()
