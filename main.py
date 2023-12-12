import csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue as p_queue

# Data parsing to create a pandas data frame and clean it up

header = ["Start", "Next", "Line", "Distance", "Zone", "Secondary Zone"]
df = pd.read_csv("tubedata.csv", names=header, skipinitialspace=True)
df.loc[342:342, 'Start':'Secondary Zone'] = "Heathrow Terminal 4", "Heathrow Terminals 1,2,3", "Piccadilly", 5, "6", "0"
df.loc[343:343, 'Start':'Secondary Zone'] = "Heathrow Terminals 1,2,3", "Hatton Cross", "Piccadilly", 3, "6", "0"
df['Zone'] = df['Zone'].replace(['a','b'],'7')
df['Secondary Zone'] = df['Secondary Zone'].replace(['a','b'],'7')
df['Zone'] = df['Zone'].replace(['c','d'],['8','9'])
df['Secondary Zone'] = df['Secondary Zone'].replace(['c','d'],['8','9'])
G = nx.from_pandas_edgelist(df, source = 'Start', target = 'Next', edge_attr=['Line', 'Distance', 'Zone', 'Secondary Zone'], create_using=nx.MultiGraph())



# Function to construct paths for DFS and BFS

def construct_path_from_root(node, root):


    path_from_root = [node['label']]
    while node['parent']:
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
    return path_from_root


#Implementation of DFS

def my_depth_first_graph_search(nxobject, initial, goal, compute_exploration_cost=False, reverse=False, time=False):
    """Calculates path using depth first search"""


    frontier = [{'label': initial, 'parent': None}]
    explored = {initial}
    number_of_explored_nodes = 1

    while frontier:
        node = frontier.pop()  # pop from the right of the list

        number_of_explored_nodes += 1
        if node['label'] == goal:
            if time:
                time_taken = calculate_time_taken(child, nxobject)
                print('time taken = {}'.format(time_taken))
            if compute_exploration_cost:
                print('number of explorations = {}'.format(number_of_explored_nodes))
            return node

        neighbours = reversed(list(nxobject.neighbors(node['label']))) if reverse else nxobject.neighbors(node['label'])
        for child_label in neighbours:
            child = {'label': child_label, 'parent': node}
            if child_label not in explored:
                frontier.append(child)  # added to the right of the list, so it is a LIFO
                explored.add(child_label)
    return None

#Implementation of BFS
def my_breadth_first_graph_search(nxobject, initial, goal, compute_exploration_cost=False, reverse=False, time=False):
    """Calculates path using depth first search"""


    if initial == goal:  # just in case, because now we are checking the children
        return None

    number_of_explored_nodes = 1
    frontier = [{'label': initial, 'parent': None}]
    # FIFO queue should NOT be implemented with a list, this is slow! better to use deque
    explored = {initial}

    while frontier:
        node = frontier.pop()  # pop from the right of the list

        neighbours = reversed(list(nxobject.neighbors(node['label']))) if reverse else nxobject.neighbors(node['label'])
        for child_label in neighbours:

            child = {'label': child_label, 'parent': node}

            if child_label == goal:
                if time:
                    time_taken = calculate_time_taken(child, nxobject)
                    print('time taken = {}'.format(time_taken))
                if compute_exploration_cost:
                    print('number of explorations = {}'.format(number_of_explored_nodes))
                return child

            if child_label not in explored:
                frontier = [child] + frontier  # added to the left of the list, so a FIFO!
                number_of_explored_nodes += 1
                explored.add(child_label)

    return None


# Calculate time function

def calculate_time_taken(node, nxobject):
    """Function to calculate the time taken in our DFS and BFS implementations"""

    path_from_root = [node['label']]
    time_taken = 0

    while node['parent']:
        time_taken += int(nxobject.get_edge_data(node['label'], node['parent']['label'])[0]['Distance'])
        node = node['parent']
        path_from_root = [node['label']] + path_from_root
    return time_taken


# Empty Queue Function

def gc(queue):
    """Function to clean garbage from the priority queue"""
    if not queue.empty():
        while not queue.empty():
            queue.get()

#Implementation of UCS
def ucs(G, start, goal, compute_exploration_cost=False, time=False):
    """
    This returns the path with the least cost from start to goal.
    This also prints out the time taken for path and number of
    explorations if those parameters are set to True.

    """
    # Make queue and make sure it's empty.
    my_queue = p_queue()
    gc(my_queue)

    cost = {}
    back_pointer = {}
    cost[start] = 0
    path = [goal]
    number_of_explored_nodes = 0


    my_queue.put((0, start))  # Cost of start node is 0
    closed = set()

    while True:
        if my_queue.empty():
            print("There is no path from {} to {}".format(start, goal), file=stderr)
            return None

        while True:

            current_cost, current = my_queue.get()

            if current not in closed:
                # Add current to the closed set
                closed.add(current)
                break

        if current == goal:
            dummy = goal
            while dummy != start:
                path.append(back_pointer[dummy])
                dummy = back_pointer[dummy]

            path.reverse()
            if time:
                print('time taken = {}'.format(current_cost))
            if compute_exploration_cost:
                print('number of explorations = {}'.format(number_of_explored_nodes))

            return path

        number_of_explored_nodes += 1
        # Add nodes adjacent to current to the my_queue
        # provided they are not in the closed set.
        if G[current]:
            for node in G[current]:
                if node not in closed:

                    node_cost = current_cost + int(G[current][node][0]['Distance']) #Cost function

                    if node not in cost or cost[node] > node_cost:

                        back_pointer[node] = current
                        cost[node] = node_cost

                    my_queue.put((node_cost, node))


solution = my_depth_first_graph_search(G, "Rayners Lane", 'Kensal Green', compute_exploration_cost=True, time=True, reverse=False)
solution2 = my_breadth_first_graph_search(G, "Rayners Lane", 'Kensal Green', compute_exploration_cost=True, time=True, reverse=True)
x = construct_path_from_root(solution, "Euston")
y = construct_path_from_root(solution2, "Euston")


print('DFS path: ', x)
print('BFS path: ', y)




#Implementation of UCS improved
def ucs_improved(G, start, goal, compute_exploration_cost=False, time=False):
    """
    This returns the path with the least cost from start to goal.
    This also prints out the time taken for path and number of
    explorations if those parameters are set to True. This function
    adds a cost for changing lines.

    """
    # Make queue and make sure it's empty.
    my_queue = p_queue()
    gc(my_queue)

    cost = {}
    back_pointer = {}
    cost[start] = 0
    path = [goal]
    number_of_explored_nodes = 0
    counter = 0
    my_queue.put((0, start))  # Cost of start node is 0
    closed = set()

    while True:
        if my_queue.empty():
            print("There is no path from {} to {}".format(start, goal), file=stderr)
            return None

        while True:

            current_cost, current = my_queue.get()

            if current not in closed:
                # Add current to the closed set
                closed.add(current)
                break

        if current == goal:
            dummy = goal
            time_taken = 0
            while dummy != start:
                path.append(back_pointer[dummy])
                time_taken += (int(G[dummy][back_pointer[dummy]][0]['Distance']))
                dummy = back_pointer[dummy]

            path.reverse()
            if time:
                print('time taken = {}'.format(time_taken))
            if compute_exploration_cost:
                print('number of explorations = {}'.format(number_of_explored_nodes))

            return path

        number_of_explored_nodes += 1
        node_cost = 0
        penalty = 100

        # Add nodes adjacent to current to the my_queue
        # provided they are not in the closed set.
        if G[current]:
            for node in G[current]:
                if node not in closed:

                    if counter != 0:

                        if G[current][node][0]['Line'] != G[previous_current][previous_node][0]['Line']:
                            node_cost = current_cost + int(G[current][node][0]['Distance']) + penalty# add cost for changing lines

                        else:
                            node_cost = current_cost + int(G[current][node][0]['Distance'])



                    if node not in cost or cost[node] > node_cost:
                        back_pointer[node] = current

                        cost[node] = node_cost
                        previous_current = current
                        previous_node = node
                    counter += 1

                    my_queue.put((node_cost, node))

#Heuristic function
def heuristic(G, node, current, goal):


    h_list = df.index[df['Start'] == goal].tolist()
    goal_zone = df.loc[int(h_list[0])]['Zone']

    heuristic_cost = (int(goal_zone) - int(G[current][node][0]['Zone']))**4

    return heuristic_cost


#Implementation of heuristic search
def heuristic_search(G, start, goal, compute_exploration_cost=False, time=False):
    """
    This returns the path with the least cost from start to goal.
    This function also adds a heuristic for leading the search
    towards stations that are closer to the zone of the goal
    station

    """
    # Make queue and make sure it's empty.
    my_queue = p_queue()
    gc(my_queue)

    cost = {}
    back_pointer = {}
    cost[start] = 0
    path = [goal]
    number_of_explored_nodes = 0
    counter = 0
    my_queue.put((0, start))  # Cost of start node is 0
    closed = set()

    while True:
        if my_queue.empty():
            print("There is no path from {} to {}".format(start, goal), file=stderr)
            return None

        while True:

            current_cost, current = my_queue.get()

            if current not in closed:
                # Add current to the closed set
                closed.add(current)
                break

        if current == goal:
            dummy = goal
            time_taken = 0
            while dummy != start:
                path.append(back_pointer[dummy])
                time_taken += (int(G[dummy][back_pointer[dummy]][0]['Distance']))
                dummy = back_pointer[dummy]

            path.reverse()
            if time:
                print('time taken = {}'.format(time_taken))
            if compute_exploration_cost:
                print('number of explorations = {}'.format(number_of_explored_nodes))

            return path

        number_of_explored_nodes += 1
        node_cost = 0
        penalty = 5

        # Add nodes adjacent to current to the my_queue
        # provided they are not in the closed set.
        if G[current]:
            for node in G[current]:
                if node not in closed:
                    if counter != 0:
                        if G[current][node][0]['Line'] != G[previous_current][previous_node][0]['Line']:
                            node_cost = current_cost + int(G[current][node][0]['Distance']) + penalty + heuristic(G, node, current, goal)

                        else:
                            node_cost = current_cost + int(G[current][node][0]['Distance']) + heuristic(G, node, current, goal)



                    if node not in cost or cost[node] > node_cost:
                        back_pointer[node] = current

                        cost[node] = node_cost
                        previous_current = current
                        previous_node = node
                    counter += 1

                    my_queue.put((node_cost, node))




k = ucs(G, "Ealing Broadway", 'South Kensington', compute_exploration_cost=True, time=True)
l = ucs_improved(G, "Ealing Broadway", 'South Kensington', compute_exploration_cost=True, time=True)
h = heuristic_search(G, "Ealing Broadway", 'South Kensington', compute_exploration_cost=True, time=True)


print("UCS path: ", k)
print("UCS improved path: ", l)
print("Heuristic path: ", h)
