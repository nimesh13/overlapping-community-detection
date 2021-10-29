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

def createLineGraph(graph, matrix):
    num_nodes = len(matrix)
    line_graph = [[0] * num_nodes for i in range(num_nodes)]
    
    for i in range(0, num_nodes):
        for j in range(0, i):
            if matrix[i][j] == 1:
                row, column = getNodes(graph, i + 1, j + 1)

                similarity = findSimilarity(row, column)
                
                line_graph[i][j] = similarity
                line_graph[j][i] = similarity

    return line_graph

def findSimilarity(nodeA, nodeB):
    a = set(nodeA)
    b = set(nodeB)
    S = 1.0 * len(a & b) / len(a | b) # Jacc similarity...
    return S

def getNodes(graph, nodeA, nodeB):
    valA = [item[1] for item in graph if item[0] == str(nodeA)]
    valB = [item[1] for item in graph if item[0] == str(nodeB)]

    return valA[0], valB[0]

number_of_vertices = 0
edges = [('1', '2'), ('1', '3'), ('2', '3'), ('3', '4'), ('3', '5'), ('4', '5')]
graph = project(edges)
# print(graph)
matrix = createAdjacencyMatrix(graph)
# print(matrix)
line_graph = createLineGraph(graph, matrix)
print(line_graph)