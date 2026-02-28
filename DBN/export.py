import networkx as nx
import csv
from typing import List
from graph import DBNGraph

class GraphExporter:
    """
    Export DBNGraph to:
      - Edge list CSV
      - GraphML
    """

    def __init__(self, genes: List[str]):
        self.genes = genes
        self.n = len(genes)

    def save_edge_list(self, graph: DBNGraph, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "lag"])
            # intra
            for i in range(self.n):
                for j in range(self.n):
                    if graph.G0[i, j]:
                        writer.writerow([self.genes[i], self.genes[j], 0])
            # inter
            for lag in range(graph.order_l):
                for i in range(self.n):
                    for j in range(self.n):
                        if graph.Gt[lag, i, j]:
                            writer.writerow([self.genes[i], self.genes[j], lag + 1])

    def save_graphml(self, graph: DBNGraph, path: str):
        G = nx.DiGraph()
        for g in self.genes:
            G.add_node(g)
        # intra
        for i in range(self.n):
            for j in range(self.n):
                if graph.G0[i, j]:
                    G.add_edge(self.genes[i], self.genes[j], lag=0)
        # inter
        for lag in range(graph.order_l):
            for i in range(self.n):
                for j in range(self.n):
                    if graph.Gt[lag, i, j]:
                        G.add_edge(self.genes[i], self.genes[j], lag=lag + 1)
        nx.write_graphml(G, path)
