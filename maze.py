import heapq
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

def generate_maze() -> list:
    # Predefined 10x10 maze
    return [
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    ]

def get_neighbors(maze, position):
    # Get valid neighboring cells (up, down, left, right)
    row, col = position
    neighbors = []
    if row > 0 and maze[row - 1][col] == 0:  # Up
        neighbors.append((row - 1, col))
    if row < len(maze) - 1 and maze[row + 1][col] == 0:  # Down
        neighbors.append((row + 1, col))
    if col > 0 and maze[row][col - 1] == 0:  # Left
        neighbors.append((row, col - 1))
    if col < len(maze[0]) - 1 and maze[row][col + 1] == 0:  # Right
        neighbors.append((row, col + 1))
    return neighbors

def dfs(maze, start, goal):
    # Depth-First Search (DFS)
    stack = [(start, [start])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex in visited:
            continue
        visited.add(vertex)
        for neighbor in get_neighbors(maze, vertex):
            if neighbor == goal:
                return path + [neighbor]
            stack.append((neighbor, path + [neighbor]))
    return None  # No path

def bfs(maze, start, goal):
    # Breadth-First Search (BFS)
    queue = deque([(start, [start])])
    visited = set()
    while queue:
        (vertex, path) = queue.popleft()
        if vertex in visited:
            continue
        visited.add(vertex)
        for neighbor in get_neighbors(maze, vertex):
            if neighbor == goal:
                return path + [neighbor]
            queue.append((neighbor, path + [neighbor]))
    return None  # No path

def ucs(maze, start, goal):
    # Uniform Cost Search (UCS)
    priority_queue = [(0, start, [start])]
    visited = set()
    while priority_queue:
        (cost, vertex, path) = heapq.heappop(priority_queue)
        if vertex in visited:
            continue
        visited.add(vertex)
        if vertex == goal:
            return path
        for neighbor in get_neighbors(maze, vertex):
            heapq.heappush(priority_queue, (cost + 1, neighbor, path + [neighbor]))
    return None  # No path

def heuristic(a, b):
    # Manhattan distance between two points
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(maze, start, goal):
    # A* Search
    priority_queue = [(0, start, [start])]
    visited = set()
    while priority_queue:
        (cost, vertex, path) = heapq.heappop(priority_queue)
        if vertex in visited:
            continue
        visited.add(vertex)
        if vertex == goal:
            return path
        for neighbor in get_neighbors(maze, vertex):
            heapq.heappush(priority_queue, (cost + 1 + heuristic(neighbor, goal), neighbor, path + [neighbor]))
    return None  # No path

def best_first_search(maze, start, goal):
    # Best-First Search
    priority_queue = [(heuristic(start, goal), start, [start])]
    visited = set()
    while priority_queue:
        (h_cost, vertex, path) = heapq.heappop(priority_queue)
        if vertex in visited:
            continue
        visited.add(vertex)
        if vertex == goal:
            return path
        for neighbor in get_neighbors(maze, vertex):
            heapq.heappush(priority_queue, (heuristic(neighbor, goal), neighbor, path + [neighbor]))
    return None  # No path

def visualize_maze(maze, path):
    # Visualize the maze and the path (red)
    maze_array = np.array(maze)
    for position in path:
        maze_array[position[0]][position[1]] = 2
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'red'])
    bounds = [0, 1, 2, 3]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(maze_array, cmap=cmap, norm=norm)
    plt.grid(False)
    plt.show()

def main():
    # Main function
    maze = generate_maze()
    start, goal = (0, 0), (9, 9)
    print("Choose an algorithm: DFS, BFS, UCS, A*, Best-First")
    choice = input().strip().lower()
    path = None
    if choice == 'dfs':
        path = dfs(maze, start, goal)
    elif choice == 'bfs':
        path = bfs(maze, start, goal)
    elif choice == 'ucs':
        path = ucs(maze, start, goal)
    elif choice == 'a*':
        path = a_star(maze, start, goal)
    elif choice == 'best-first':
        path = best_first_search(maze, start, goal)
    else:
        print("Invalid choice!")
        return
    
    if path:
        print(f"Path found: {path}")
        visualize_maze(maze, path)
    else:
        print("No path found.")

if __name__ == "__main__":
    main()
