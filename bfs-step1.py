# Adapted from: 
# https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
#
# Python3 Program to print BFS traversal
# from a given source vertex. BFS(int s)
# traverses vertices reachable from s.

# This class represents a directed graph
# using adjacency list representation

# In[1]:
class Graph:

    # Constructor
    def __init__(self):
        # default dictionary to store graph
        self.graph = {}

    # function to add an edge to graph
    def add_edge(self, u, v):
        if not u in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def set_goal(self, goal: int):
        self.goal = goal
        print('Goal is %s' % self.goal)

    def is_goal(self,v: int):
        return v == self.goal

    def print_best_route(self):
        print(self.goal, end=" ")
        cur = self.best_route[self.goal]
        while cur is not None:
            print("<- %s" % cur, end=" ")
            cur = self.best_route[cur]

    # Function to print a BFS of graph
    def bfs(self, s):

        # Mark all the vertices as not visited
        visited = [False] * (len(self.graph))

        # Create a queue for BFS
        queue = []

        #Best Route
        self.best_route = {}
        # Mark the source node as
        # visited and enqueue it
        queue.append(s)
        visited[s] = True
        self.best_route[s] = None

        while queue:

            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0)
            print(s, end=" ")

            if self.is_goal(s):
                print("[GOAL FOUND!]")
                self.print_best_route()
                break

            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
                    self.best_route[i] = s

# In[2]:
# Driver code

# Create a graph given in
# the above diagram
g = Graph()
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)
g.add_edge(2, 3)
g.add_edge(3, 3)

g.set_goal(3)

print("Data structure: ", g.graph)

# In[3]:
print("Following is Breadth First Traversal"
      " (starting from vertex 2)")
g.bfs(2)

# This code is contributed by Neelam Yadav
