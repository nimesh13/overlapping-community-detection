from collections import defaultdict

def project(edges, nodeType=int):
    graph = defaultdict(list)
    for edge in edges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

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

    line_graph = [[-1] * num_lines for i in range(num_lines)]

    for i in range(0, num_nodes):
        nodeHub = i + 1
        adjCombList = getAdjCombinationOf2(graph, nodeHub)

        for comb in range(0, len(adjCombList)):
            nodeA = adjCombList[comb][0]
            nodeB = adjCombList[comb][1]

            nodeA_and_neighbors, nodeB_and_neighbors = getNodes(graph, nodeA, nodeB)
            nodeA_and_neighbors.append(nodeHub)
            nodeB_and_neighbors.append(nodeHub)

            similarity = findSimilarity(nodeA_and_neighbors, nodeB_and_neighbors)
            nodeA_id = edgeIdMatrix[nodeHub-1][nodeA-1]
            nodeB_id = edgeIdMatrix[nodeHub-1][nodeB-1]

            line_graph[nodeA_id][nodeB_id] = similarity
            line_graph[nodeB_id][nodeA_id] = similarity


        #for j in range(0, i):
        #    if matrix[i][j] == 1:
        #        row, column = getNodes(graph, i + 1, j + 1)
                #print(i+1,row, j+1,column)

        #       similarity = findSimilarity(row, column)

        #        line_graph[i][j] = similarity
        #        line_graph[j][i] = similarity

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


number_of_vertices = 0
edges = [('1', '2'), ('1', '3'), ('1', '4'), ('2', '3'), ('2', '4'), ('3', '4'), ('4', '5'), ('4' ,'6'), ('4', '7'), ('5', '6'), ('7', '8'), ('7','9'), ('8', '9')]
graph = project(edges)
print(graph)

matrix = createAdjacencyMatrix(graph)
for entry in range(0, len(matrix)):
    print(matrix[entry])

edgeIdMx = createEdgeIdMatrix(matrix)
for entry in range(0, len(edgeIdMx)):
    print(edgeIdMx[entry])

line_graph = createLineGraph(graph, matrix, edgeIdMx)

for entry in range(0, len(line_graph)):
    print(line_graph[entry])