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

    line_graph = py.zeros(shape=(num_lines, num_lines), dtype=py.float32)

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

def partition_density(communityId, communities, nodes_belongs_to):
    m_c = 0.0
    for key in communities:
        if communities[key] == communityId:
            m_c += 1.0

    n_c = 0.0
    for entry in nodes_belongs_to:
        for item in entry[1]:
            if item == communityId:
                n_c += 1.0


    denom = (((n_c*(n_c-1.0))/2.0) - (n_c-1.0))
    if denom > 0:
        Dc = (m_c - (n_c - 1.0))/denom
        print("partition densitiy of community[%d] is  %f" % (communityId,Dc))
    else:
        print("partition densitiy of community[%d] is none" % communityId)


def average_partition_density(communities, distinct_com, nodes_belongs_to):
    numCom = len(distinct_com)
    numLink = len(communities)
    m_c = {}
    n_c = {}
    for key in communities:
        comId = communities[key]
        m_c[comId] = 0.0
        n_c[comId] = 0.0

    for key in communities:
        comId = communities[key]
        linkC = m_c[comId]
        m_c[comId] = linkC + 1.0

    for entry in nodes_belongs_to:
        for item in entry[1]:
            nodeC = n_c[item]
            n_c[item] = nodeC + 1.0

    sum = 0.0
    for i in range(0, numCom):
        denom = ((n_c[i]-2.0)*(n_c[i]-1.0))
        if denom > 0:
            sum += m_c[i]*((m_c[i] - (n_c[i] - 1.0))/denom)
    
    D = (2.0/numLink)*sum
    print("Average Partition density is", D)


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

for i in range(0, len(distinct_communities)):
    partition_density(i, communities, original_graph)

average_partition_density(communities, distinct_communities, original_graph)


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


