import pygame as pg
from os import path
import heapq
vec = pg.math.Vector2
import math

# Todo: Create a grid for a given size of environment
# Todo: Should be a parameter of the environment
TILESIZE = 32
GRIDWIDTH = 40
GRIDHEIGHT = 20
WIDTH = TILESIZE * GRIDWIDTH
HEIGHT = TILESIZE * GRIDHEIGHT

DARKGRAY = (40, 40, 40)
MEDGRAY = (75, 75, 75)
LIGHTGRAY = (140, 140, 140)
SAFETYPADDINGCOLOR = (204, 162, 10)

pg.init()
screen = pg.display.set_mode((WIDTH, HEIGHT))



class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = []
        self.connections = [vec(1, 0), vec(-1, 0), vec(0, 1), vec(0, -1)] #possible movements:  right, left, up and down
        self.connections += [vec(1, 1), vec(-1, 1), vec(1, -1), vec(-1, -1)]
        self.weights = {} #key is grid location, value is cost --  {(1,1):10}

    def in_bounds(self, node): #check that node is not outside (width and height) of the map
        return 0 <= node.x < self.width and 0 <= node.y < self.height

    def passable(self, node):
        return node not in self.obstacles

    def find_neighbors(self, node):
        neighbors = [node + connection for connection in self.connections]
        neighbors = filter(self.in_bounds, neighbors)

        neighbors = filter(self.passable, neighbors)
        return neighbors


class Environment(Grid):
    def __init__(self, width, height):
        super().__init__(width, height)
        #self.weights = {}

    def cost(self, from_node, to_node):
        # ToDo: Costs should be parameters of the Gridbased-PathPlanning Algorithms. Environments only have sizes,
        #  obstacles and agvs
        if (vec(to_node) - vec(from_node)).length_squared() == 1:#if distance is 1, we're moving vertically or horizontally
            return self.weights.get(to_node, 0) + 10   #return 0 if it's no in the weights list -- if there' no weigh assigned to it, the default weight will be 0
        else:
            return self.weights.get(to_node, 0) + 14

    def draw(self):
        #Todo: Drawing should be done outside of the class given an environment
        #draw obstacles:
        for obstacle in self.obstacles:
            rect = pg.Rect(obstacle * TILESIZE, (TILESIZE, TILESIZE))
            pg.draw.rect(screen, LIGHTGRAY, rect)

        #draw safety padding:
        for tile in self.weights:
            x, y = tile
            rect = pg.Rect(x * TILESIZE + 3, y * TILESIZE + 3, TILESIZE - 5, TILESIZE - 5)
            pg.draw.rect(screen, SAFETYPADDINGCOLOR, rect)

    def add_cost_to_safety_padding_NO_corners(self, safetyPaddingNoCorners):
        self.safetyPaddingNoCorners = safetyPaddingNoCorners
        for tile in safetyPaddingNoCorners:
            self.weights[tile] = 16

    def add_cost_to_safety_padding_corners(self, safetyPaddingCorners):
        self.safetyPaddingCorners = safetyPaddingCorners
        for tile in safetyPaddingCorners:
            self.weights.update({tile: 9})


class PriorityQueue:
    #heapq binary tree

    def __init__(self):
        self.nodes = []

    #add to heap
    def put(self, node, cost):
        heapq.heappush(self.nodes, (cost, node))

    #remove from heap
    def get(self):
        return heapq.heappop(self.nodes)[1]

    def empty(self):
        return len(self.nodes) == 0  #if = 0, return True


class PathPlanning:
    # Todo: in the general PathPlanning class we need a method called get_path(from,to) that returns a path
    #  The A*star should than be derived from this class
    # ToDo: maybe certain specialities of the environment (like the costs and the grid sizes) are part of a certain
    #  planning class and should be integrated here
    #calculate path:

    def heuristic(self, node1, node2):
        #use Euclidian distance calculation:
        return (math.sqrt(abs(node1.x - node2.x)** 2 + abs(node1.y - node2.y)** 2)) * 10

    def a_star_search(self, graph, start, end):
        frontier = PriorityQueue()
        frontier.put(vec2int(start), 0)
        path = {}
        cost = {}
        path[vec2int(start)] = None
        cost[vec2int(start)] = 0

        while not frontier.empty():
            current = frontier.get()
            if current == end:
                break
            for nextTile in graph.find_neighbors(vec(current)):
                nextTile = vec2int(nextTile)
                next_cost = cost[current] + graph.cost(current, nextTile)
                if nextTile not in cost or next_cost < cost[nextTile]:
                    cost[nextTile] = next_cost
                    priority = next_cost + self.heuristic(end, vec(nextTile))
                    frontier.put(nextTile, priority)
                    path[nextTile] = vec(current) - vec(nextTile)
        return path, cost


    #draw path with arrows:

    def draw_path_arrows(self, path, start, goal):
        # Todo: drawing outside of the class
        arrows = {}
        arrow_img = pg.image.load('arrowRight.png').convert_alpha()
        arrow_img = pg.transform.scale(arrow_img, (12, 12))

        for direction in [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            arrows[direction] = pg.transform.rotate(arrow_img, vec(direction).angle_to(vec(1, 0)))

        current = start # + path[vec2int(start)]
        l = 0
        while current != goal:
            v = path[(current.x, current.y)]
            if v.length_squared() == 1:
                l += 10
            else:
                l += 14
            img = arrows[vec2int(v)]
            x = current.x * TILESIZE + TILESIZE / 2
            y = current.y * TILESIZE + TILESIZE / 2
            r = img.get_rect(center=(x, y))
            screen.blit(img, r)
            current = current + path[vec2int(current)]


class AGV:
    # ToDo: AGVs have a position, a direction and a size. A list of agvs should be part of the environment
    # ToDo: transport orders are given to the AGV (from, to)
    # ToDo: The AGV uses a planning algorithm to find the route to the next target
    def __init__(self, start, goal, path_planning: PathPlanning = None):

        self.start = start
        self.goal = goal
        self.path_planning: PathPlanning = path_planning

    def draw_icons(self):
        # ToDo: Drawing outside of the class
        #icon_dir = path.join(path.dirname(__file__), '../images')
        start_img = pg.image.load('start.png').convert_alpha()
        start_img = pg.transform.scale(start_img, (25, 25))
        start_img.fill((0, 255, 0, 255), special_flags=pg.BLEND_RGBA_MULT)
        workstation_img = pg.image.load('workstation.png').convert_alpha()
        workstation_img = pg.transform.scale(workstation_img, (25, 25))
        workstation_img.fill((255, 0, 0, 255), special_flags=pg.BLEND_RGBA_MULT)

        start_center = (self.start.x * TILESIZE + TILESIZE / 2, self.start.y * TILESIZE + TILESIZE / 2)
        screen.blit(start_img, start_img.get_rect(center=start_center))
        goal_center = (self.goal.x * TILESIZE + TILESIZE / 2, self.goal.y * TILESIZE + TILESIZE / 2)
        screen.blit(workstation_img, workstation_img.get_rect(center=goal_center))



class DrawingTool:
    pass

def draw_grid():
    for x in range(0, WIDTH, TILESIZE):
        pg.draw.line(screen, LIGHTGRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, TILESIZE):
        pg.draw.line(screen, LIGHTGRAY, (0, y), (WIDTH, y))


def vec2int(v):
    return (int(v.x), int(v.y))


e = Environment(GRIDWIDTH, GRIDHEIGHT)


#draw obstacles
obstacles = [(5, 9), (5, 10), (5, 11), (5, 12), (5, 13), (4, 13), (4, 12), (4, 11), (4, 10), (4, 9), (3, 5), (3, 4), (4, 4), (4, 5), (4, 14), (5, 14), (17, 3), (17, 4), (17, 5), (18, 5), (18, 4), (18, 3), (17, 6), (18, 6), (19, 3), (19, 4), (19, 5), (19, 6), (23, 3), (23, 4), (23, 5), (23, 6), (24, 6), (25, 6), (25, 5), (24, 5), (24, 4), (25, 4), (25, 3), (24, 3), (17, 11), (17, 12), (17, 13), (18, 13), (19, 13), (19, 12), (19, 11), (18, 11), (18, 12), (23, 11), (23, 12), (23, 13), (25, 13), (24, 13), (25, 12), (24, 12), (24, 10), (25, 10), (25, 11), (24, 11), (5, 4), (5, 5), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14), (13, 6), (12, 6), (11, 6), (13, 3), (12, 3), (11, 3), (11, 4), (11, 5), (13, 5), (13, 4), (12, 4), (12, 5), (11, 11), (11, 12), (11, 13), (13, 13), (12, 13), (12, 12), (13, 12), (13, 11), (12, 11), (23, 10), (19, 10), (18, 10), (17, 10), (13, 10), (12, 10), (11, 10)]
for obstacle in obstacles:
    e.obstacles.append(vec(obstacle))

#draw padding ('padding' and 'corner padding' have different weights)

# ToDo: these corners should be auto-calculated within the environement
safetyPaddingNoCorners = [(11, 9), (13, 9), (12, 9), (14, 10), (14, 11), (14, 12), (14, 13), (12, 14), (13, 14), (10, 13), (10, 12), (10, 11), (10, 10),(16, 13), (16, 12), (16, 11), (16, 10), (17, 9), (18, 9), (19, 9), (20, 10), (20, 11), (20, 12), (20, 13), (19, 14), (18, 14), (22, 13), (22, 12), (22, 11), (22, 10), (23, 9), (24, 9), (25, 9), (26, 10), (26, 11), (26, 12), (26, 13), (25, 14), (24, 14), (25, 7), (24, 7), (22, 6), (22, 5), (22, 4), (22, 3), (23, 2), (24, 2), (25, 2), (26, 3), (26, 4), (26, 5), (26, 6), (19, 7), (18, 7), (20, 6), (20, 5), (20, 4), (20, 3), (19, 2), (18, 2), (17, 2), (16, 3), (16, 4), (16, 5), (16, 6), (13, 7), (12, 7), (14, 6), (14, 5), (14, 4), (14, 3), (13, 2), (12, 2), (11, 2), (10, 3), (10, 4), (10, 5), (10, 6), (5, 3), (4, 3), (3, 3), (2, 4), (2, 5), (3, 6), (4, 6), (5, 6), (6, 4), (4, 8), (5, 8), (3, 9), (3, 11), (3, 10), (3, 12), (3, 13), (3, 14), (4, 15), (5, 15), (6, 15), (7, 14), (7, 13), (7, 12), (7, 11), (7, 10), (7, 9)]
e.add_cost_to_safety_padding_NO_corners(safetyPaddingNoCorners)

safetyPaddingCorners = [(6, 6), (2, 6), (2, 3), (6, 3), (10, 2), (14, 2), (14, 7), (10, 7), (16, 2), (16, 7), (20, 7), (20, 2), (22, 2), (22, 7), (26, 2), (26, 7), (22, 9), (26, 9), (26, 14), (22, 14), (16, 9), (20, 9), (20, 14), (16, 14), (10, 9), (14, 9), (14, 14), (10, 14), (3, 8), (7, 8), (7, 15), (3, 15)]
e.add_cost_to_safety_padding_corners(safetyPaddingCorners)




agv1 = AGV(vec(12, 15), vec(1, 1))
p = PathPlanning()
agv1path, agv1cost = p.a_star_search(e, agv1.goal, agv1.start)



running = True
while running:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_ESCAPE:
                running = False
            if event.key == pg.K_m:
                # print the obstacle coordinates list for saving
                print([(int(loc.x), int(loc.y)) for loc in e.obstacles])

        #add/remove obstacles(move to DrawingTool later):
        if event.type == pg.MOUSEBUTTONDOWN:
            mpos = vec(pg.mouse.get_pos()) // TILESIZE
            if event.button == 1:
                if mpos in e.obstacles:
                    e.obstacles.remove(mpos)
                else:
                    e.obstacles.append(mpos)
    screen.fill(DARKGRAY)
    draw_grid()
    p.draw_path_arrows(agv1path, agv1.start, agv1.goal)
    e.draw()
    agv1.draw_icons()
    pg.display.flip()
pg.quit()



