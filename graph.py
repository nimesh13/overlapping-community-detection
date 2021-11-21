from collections import defaultdict
import argparse
from os import PRIO_USER
import community as community_louvain
import networkx as nx
import numpy as py
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def project(filename, delimiter=' ', nodeType=int):
    graph = defaultdict(list)
    for line in open(filename):
        L = line.strip().split(delimiter)
        if int(L[1]) not in graph[int(L[0])] and L[0] != L[1]:
            graph[int(L[0])].append(int(L[1]))
            graph[int(L[1])].append(int(L[0]))

    return sorted(graph.items())

def createLineDict(graphDict):
    lineDict = {}
    revertLineDict = {}
    lineId = 0
    for entry in graphDict:
        for item in entry[1]:
            left = 0
            right = 0
            if (entry[0] < item):
                left = entry[0]
                right = item
            elif (item < entry[0]):
                left = item
                right = entry[0]

            if revertLineDict.get((left, right)) == None:
                lineDict[lineId] = (left, right)
                revertLineDict[(left, right)] = lineId
                lineId += 1

    return lineDict, revertLineDict

def createLineGraph(graph, lineDict, revertLineDict):

    num_lines = len(lineDict)
    num_nodes = len(graph)

    line_graph = [[0] * num_lines for i in range(num_lines)]

    for entry in graph:
        nodeHub = int(entry[0])
        adjCombList = getAdjCombinationOf2(entry[1])

        for comb in range(0, len(adjCombList)):
            nodeA = adjCombList[comb][0]
            nodeB = adjCombList[comb][1]

            nodeA_and_neighbors, nodeB_and_neighbors = getNodes(graph, nodeA, nodeB)
            nodeA_and_neighbors.append(nodeA)
            nodeB_and_neighbors.append(nodeB)

            #print(nodeHub, ":", nodeA ,nodeA_and_neighbors, "," ,nodeB ,nodeB_and_neighbors)

            similarity = findSimilarity(nodeA_and_neighbors, nodeB_and_neighbors)
            lineA_id = revertLineDict[(nodeHub, nodeA)] if nodeHub < nodeA else revertLineDict[(nodeA, nodeHub)] 
            lineB_id = revertLineDict[(nodeHub, nodeB)] if nodeHub < nodeB else revertLineDict[(nodeB, nodeHub)]   

            line_graph[lineA_id][lineB_id] = similarity
            line_graph[lineB_id][lineA_id] = similarity

    return line_graph

def findSimilarity(nodeA, nodeB):
    a = set(nodeA)
    b = set(nodeB)
    S = 1.0 * len(a & b) / len(a | b)
    return S

def getNodes(graph, nodeA, nodeB):
    valA = [item[1] for item in graph if item[0] == nodeA]
    valB = [item[1] for item in graph if item[0] == nodeB]
    adjA = valA[0][:]
    adjB = valB[0][:]
    return adjA, adjB

def getAdjCombinationOf2(nodeList):
    # Find the edge that shared the same node to compare.
    # For example, edge u->v and edge s->v shared the same node, then v is acted as the hub node
    # This function wants to find all combinations of 2 nodes (u and s) that are adjacent to hub node (v)
    # to represents the combinations of two edges.
    result = []
    numNodes = len(nodeList)
    for i in range (0, numNodes-1):
        for j in range(i+1, numNodes):
            comb = []
            if (nodeList[i] < nodeList[j]):
                comb.append(nodeList[i])
                comb.append(nodeList[j])
                result.append(comb)
            elif (nodeList[j] < nodeList[i]):
                comb.append(nodeList[j])
                comb.append(nodeList[i])
                result.append(comb)
    return result

parser = argparse.ArgumentParser()
parser.add_argument(dest='filename', help="Filename of the dataset containing the graph")

# Parse and print the results
filename = parser.parse_args().filename

number_of_vertices = 0

graph = project(filename)
print('\nThe nodes with their neighbours are: \n')
for node in graph:
    print("%s -> %s" % (node[0], node[1]))

lineDirectory, revertLineDirectory = createLineDict(graph)
#for i in lineDirectory:
#    print(i, lineDirectory[i])
#for i in revertLineDirectory:
#    print(i, revertLineDirectory[i])

line_graph = createLineGraph(graph, lineDirectory, revertLineDirectory)
print('\nThe line graph adjacency matrix with weights as similarity is: \n')
#for entry in range(0, len(line_graph)):
#    print(line_graph[entry])

# print(py.matrix(line_graph))
G = nx.Graph(py.matrix(line_graph))

communities = community_louvain.best_partition(G, weight='weight')
modularity2 = community_louvain .modularity(communities, G, weight='weight')

print("The modularity Q based on networkx is {}".format(modularity2))

distinct_communities = set(communities.values())

print('The number of distinct communities found in the graph is {}'.format(len(distinct_communities)))

original_graph = defaultdict(list)

for key in communities:
    community = communities[key]
    edge = lineDirectory[key]

    if community not in original_graph[edge[0]]:
        original_graph[edge[0]].append(community)
    if community not in original_graph[edge[1]]:
        original_graph[edge[1]].append(community)

original_graph = sorted(original_graph.items())

for node in original_graph:
    print("%s -> %s" % (node[0], sorted(node[1])))

# for node in original_graph:
#     print("%d-> %d" % (node[0], node[1]))
# pos = nx.spring_layout(K)
# cmap = cm.get_cmap('viridis', max(K.values()) + 1)
# nx.draw_networkx_nodes(G, pos, K.keys(), node_size=4,
#                        cmap=cmap, node_color=list(K.values()), label = True)
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# # print('Labels: ', list(K.keys()))
# # labels = 
# # nx.draw_networkx_labels(G, pos, K, font_size=22, font_color="whitesmoke")
# plt.show()


