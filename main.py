from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import graphviz

class EdgeType(Enum):
    directed = 1
    undirected = 2


class Vertex:
    data: Any
    index: int

    def __init__(self, data: Any, index: int):
        self.data = data
        self.index = index


class Edge:
    source: Vertex
    destination: Vertex
    weight: Optional[float]

    def __init__(self, source: Vertex, destination: Vertex, weight=None):
        self.source = source
        self.destination = destination
        self.weight = weight


class Graph:
    adjacencies: Dict[Vertex, List[Edge]]

    def __init__(self):
        self.adjacencies = {}

    def create_vertex(self, data: Any) -> Vertex:
        temp: Vertex = Vertex(data, len(self.adjacencies))
        self.adjacencies[temp] = []
        return temp

    def add_directed_edge(self, source: Vertex, destination: Vertex, weight: Optional[float] = None) -> None:
        new_edge: Edge = Edge(source, destination, 1, weight)
        self.adjacencies[source].append(new_edge)

    def add_undirected_edge(self, source: Vertex, destination: Vertex, weight: Optional[float] = None) -> None:
        new_edge = Edge(source, destination, weight)
        new_edge_revers = Edge(destination, source, weight)
        self.adjacencies[source].append(new_edge)
        self.adjacencies[destination].append(new_edge_revers)

    def add(self, edge: EdgeType, source: Vertex, destination: Vertex, weight: Optional[float] = None) -> None:
        if edge == 2:
            self.add_undirected_edge(source, destination, weight)
            self.add_undirected_edge(destination, source, weight)

def mutual_friends(g: Graph, f0: Any, f1: Any) -> List[Any]:
    vertex1 = None;
    vertex2 = None;
    neighbours = []
    for elem in list(g.adjacencies.keys()):
        if elem.data == f0:
            vertex1 = elem
        if elem.data == f1:
            vertex2 = elem
    for elem in g.adjacencies[vertex1]:
        for elem1 in g.adjacencies[vertex2]:
            if elem.destination == elem1.destination:
                neighbours.append(elem.destination.data)

    return neighbours


graph: Graph = Graph()
graph2: Graph = Graph()
graph3: Graph = Graph()
vert1 = graph.create_vertex("VI")
vert2 = graph.create_vertex("RU")
vert3 = graph.create_vertex("PA")
vert4 = graph.create_vertex("CO")
vert5 = graph.create_vertex("CH")
vert6 = graph.create_vertex("SU")
vert7 = graph.create_vertex("KE")
vert8 = graph.create_vertex("RA")
graph.add_undirected_edge(vert1, vert5)
graph.add_undirected_edge(vert1, vert2)
graph.add_undirected_edge(vert1, vert3)
graph.add_undirected_edge(vert2, vert8)
graph.add_undirected_edge(vert2, vert6)
graph.add_undirected_edge(vert3, vert4)
graph.add_undirected_edge(vert3, vert7)
graph.add_undirected_edge(vert4, vert2)
graph.add_undirected_edge(vert1, vert4)
print(mutual_friends(graph, "CO", "SU"))

vert1 = graph2.create_vertex("1")
vert2 = graph2.create_vertex("2")
vert3 = graph2.create_vertex("3")
vert4 = graph2.create_vertex("4")
vert5 = graph2.create_vertex("5")
vert6 = graph2.create_vertex("6")
vert7 = graph2.create_vertex("7")
vert8 = graph2.create_vertex("8")
graph2.add_undirected_edge(vert1, vert2)
graph2.add_undirected_edge(vert1, vert3)
graph2.add_undirected_edge(vert1, vert4)
graph2.add_undirected_edge(vert2, vert7)
graph2.add_undirected_edge(vert2, vert5)
graph2.add_undirected_edge(vert3, vert4)
graph2.add_undirected_edge(vert3, vert7)
graph2.add_undirected_edge(vert3, vert8)
graph2.add_undirected_edge(vert4, vert2)
print(mutual_friends(graph2, "1", "3"))

vert1 = graph3.create_vertex("A")
vert2 = graph3.create_vertex("B")
vert3 = graph3.create_vertex("C")
vert4 = graph3.create_vertex("D")
vert5 = graph3.create_vertex("E")
vert6 = graph3.create_vertex("F")
graph3.add_undirected_edge(vert1, vert2)
graph3.add_undirected_edge(vert1, vert3)
graph3.add_undirected_edge(vert1, vert4)
graph3.add_undirected_edge(vert1, vert5)
graph3.add_undirected_edge(vert1, vert6)
graph3.add_undirected_edge(vert2, vert3)
graph3.add_undirected_edge(vert2, vert6)
graph3.add_undirected_edge(vert3, vert4)
graph3.add_undirected_edge(vert4, vert5)
graph3.add_undirected_edge(vert5, vert6)
print(mutual_friends(graph3, "A", "B"))


graph1 = graphviz.Graph("Projekt", filename='Projekt.gv')
graph1.node("vert1", "VI")
graph1.node("vert2", "RU")
graph1.node("vert3", "PA")
graph1.node("vert4", "CH")
graph1.node("vert5", "SU")
graph1.node("vert6", "KE")
graph1.node("vert7", "RA")
graph1.edge("VI", "CH")
graph1.edge("VI", "RU")
graph1.edge("VI", "PA")
graph1.edge("RU", "RA")
graph1.edge("RU", "SU")
graph1.edge("PA", "CO")
graph1.edge("PA", "KE")
graph1.edge("CO", "RU")
graph1.edge("VI", "CO")
graph1.view()

graph2 = graphviz.Graph("Przyklad2", filename='Przyklad2.gv')
graph2.node("v1", "1")
graph2.node("v2", "2")
graph2.node("v3", "3")
graph2.node("v4", "4")
graph2.node("v5", "5")
graph2.node("v6", "6")
graph2.node("v7", "7")
graph2.node("v8", "8")
graph2.edge("1", "2")
graph2.edge("1", "3")
graph2.edge("1", "4")
graph2.edge("2", "5")
graph2.edge("2", "7")
graph2.edge("3", "4")
graph2.edge("3", "7")
graph2.edge("3", "8")
graph2.edge("4", "2")
#graph2.view()

graph3 = graphviz.Graph("Przyklad3", filename='Przyklad3.gv')
graph3.node("v1", "A")
graph3.node("v2", "B")
graph3.node("v3", "C")
graph3.node("v4", "D")
graph3.node("v5", "E")
graph3.node("v6", "F")
graph3.edge("A", "B")
graph3.edge("A", "C")
graph3.edge("A", "D")
graph3.edge("A", "E")
graph3.edge("A", "F")
graph3.edge("B", "C")
graph3.edge("C", "D")
graph3.edge("D", "E")
graph3.edge("E", "F")
graph3.edge("F", "B")
#graph3.view()



