# 1.Vacuum cleaner
a=int(input("enter no of rooms"))
for i in range(a):
    status=input()
    b=status.lower()
    print("the condition of room "+str(i)+" is "+b)
    if b=="dirty":
        print("the room is cleaning")
        if i==(a-1):
            print("the room is cleaned")
        else:
            print("room is cleaned moving to next location")
    else:
        if i==(a-1):
            print("all locations are cleaned")
        else:
            print("moving to next location")
print("vaccum cleaner task is completed")

#2.Missionaries and cannibals
print("\n")
print("\tGame Start\nNow the task is to move all of them to right side of the river")
print("rules:\n1. The boat can carry at most two people\n2. If cannibals num greater then missionaries then the cannibals would eat the missionaries\n3. The boat cannot cross the river by itself with no people on board")
lM = 3            
lC = 3            
rM=0            
rC=0            
userM = 0     
userC = 0         
k = 0
print("\nM M M C C C |     --- | \n")
try:
    while(True):
        while(True):
            print("Left side -> right side river travel")
            uM = int(input("Enter number of Missionaries travel => "))   
            uC = int(input("Enter number of Cannibals travel => "))
 
            if((uM==0)and(uC==0)):
                print("Empty travel not possible")
                print("Re-enter : ")
            elif(((uM+uC) <= 2)and((lM-uM)>=0)and((lC-uC)>=0)):
                break
            else:
                print("Wrong input re-enter : ")
        lM = (lM-uM)
        lC = (lC-uC)
        rM += uM
        rC += uC
 
        print("\n")
        for i in range(0,lM):
            print("M ",end="")
        for i in range(0,lC):
            print("C ",end="")
        print("| --> | ",end="")
        for i in range(0,rM):
            print("M ",end="")
        for i in range(0,rC):
            print("C ",end="")
        print("\n")
 
        k +=1
 
        if(((lC==3)and (lM == 1))or((lC==3)and(lM==2))or((lC==2)and(lM==1))or((rC==3)and (rM == 1))or((rC==3)and(rM==2))or((rC==2)and(rM==1))):
            print("Cannibals eat missionaries:\nYou lost the game")
 
            break
 
        if((rM+rC) == 6):
            print("You won the game : \n\tCongrats")
            print("Total attempt")
            print(k)
            break
        while(True):
            print("Right side -> Left side river travel")
            userM = int(input("Enter number of Missionaries travel => "))
            userC = int(input("Enter number of Cannibals travel => "))
             
            if((userM==0)and(userC==0)):
                    print("Empty travel not possible")
                    print("Re-enter : ")
            elif(((userM+userC) <= 2)and((rM-userM)>=0)and((rC-userC)>=0)):
                break
            else:
                print("Wrong input re-enter : ")
        lM += userM
        lC += userC
        rM -= userM
        rC -= userC
 
        k +=1
        print("\n")
        for i in range(0,lM):
            print("M ",end="")
        for i in range(0,lC):
            print("C ",end="")
        print("| <-- | ",end="")
        for i in range(0,rM):
            print("M ",end="")
        for i in range(0,rC):
            print("C ",end="")
        print("\n")
 
     
 
        if(((lC==3)and (lM == 1))or((lC==3)and(lM==2))or((lC==2)and(lM==1))or((rC==3)and (rM == 1))or((rC==3)and(rM==2))or((rC==2)and(rM==1))):
            print("Cannibals eat missionaries:\nYou lost the game")
            break
except EOFError as e:
print("\nInvalid input please retry !!")

#3.Water jug
## jug   print("there are two water jugs ")
print("j1 has 4 liters of water capacity")
print("j2 has 3 liters of water capacity")
j1=0
j2=0
print("At starting stage j1 has 0 ltrs")
print("At starting stage j2 has 0 ltrs")
a=int(input("enter the no.of ltrs in j1 : "))
if a==4:
    print("j1 has filled")
    print("Now pour the water into j2")
    j2=3
else:
    print("j1 has not filled")
a=a-j2
print("j1 has ",a,"ltr")
print("Now clear the water in j2 ")
j2=j2-3
print("j2 has ",j2,"ltrs")
print("Again pour the 1ltr of  water in j2 from j1")
a=a-1
print("j1 has ",a,"ltr")
print("j2 has 1 ltr")
b=int(input("Fill the j1:"))
if b==4:
    print("j1 has filled")
    print("pour the water in j2")
    j2=1
    j2=j2+2
    print("j2 has ",j2,"ltrs")
    b=b-2
print("J1 has",b,"ltrs of water ")

#Farmer goat wolf cabbage
print("farmer wants to cross a river with goat,wolf,cabbage")
print("The boat can carry only two persons one is farmer and the other would be a wolf or a goat or a cabbage")
print("W,C,G,F----0 to move from left to right")
print("give the input of wolf or goat or cabbage to move from left to right with farmer")
a=input()
if (a=="goat"):
    print("W,C-----G,F")
    print("input to move from right to left with farmer")
    a=input()
    if (a=="none"):
        print("W,C,F----G")
        print("input of wolf or cabbage to move from left to right with farmer")
        a=input()
        if a=="wolf":
            print("C-----F,W,G")
            print("input of wolf or goat to move from right to left with farmer")
            a=input()
            if a=="goat":
                print("G,C,F---W")
                print("input of goat or cabbage to move from left to right with farmer")
                a=input()
                if a=="cabbage":
                    print("G----F,C,W")
                    print("input of wolf or goat to move from right to left with farmer")
                    a=input()
                    if a=="none":
                        print("G,F---C,W")
                        print("do you want to move goat with farmer(yes/no)")
                        a=input()
                        if a=="yes":
                            print("0----G,F,C,W")
                            print("you won the game")
                        else:
                            print("move the goat to complete the game")
                    else:
                        print("wrong input")
                else:
                    print("wolf eats goat you lost the game")
            else:
                print("goat eats cabbage you lost the game")
        else:
            print("wrong input")
           
    else:
        print("wrong input")
elif a=="wolf":
    print("goat eats cabbage you lost the game")
else:
    print("wolf eats goat you lost the game")

# BFS algo implement
# A Node class for GBFS Pathfinding
class Node:
    def __init__(self, v, weight):
        self.v=v
        self.weight=weight

# pathNode class will help to store
# the path from src to dest.
class pathNode:
    def __init__(self, node, parent):
        self.node=node
        self.parent=parent

# Function to add edge in the graph.
def addEdge(u, v, weight):
    # Add edge u -> v with weight weight.
    adj[u].append(Node(v, weight))


# Declaring the adjacency list
adj = []
# Greedy best first search algorithm function
def GBFS(h, V, src, dest):
    """ 
    This function returns a list of 
    integers that denote the shortest
    path found using the GBFS algorithm.
    If no path exists from src to dest, we will return an empty list.
    """
    # Initializing openList and closeList.
    openList = []
    closeList = []

    # Inserting src in openList.
    openList.append(pathNode(src, None))

    # Iterating while the openList 
    # is not empty.
    while (openList):

        currentNode = openList[0]
        currentIndex = 0
        # Finding the node with the least 'h' value
        for i in range(len(openList)):
            if(h[openList[i].node] < h[currentNode.node]):
                currentNode = openList[i]
                currentIndex = i

        # Removing the currentNode from 
        # the openList and adding it in 
        # the closeList.
        openList.pop(currentIndex)
        closeList.append(currentNode)
        
        # If we have reached the destination node.
        if(currentNode.node == dest):
            # Initializing the 'path' list. 
            path = []
            cur = currentNode

            # Adding all the nodes in the 
            # path list through which we have
            # reached to dest.
            while(cur != None):
                path.append(cur.node)
                cur = cur.parent
            

            # Reversing the path, because
            # currently it denotes path
            # from dest to src.
            path.reverse()
            return path
        

        # Iterating over adjacents of 'currentNode'
        # and adding them to openList if 
        # they are neither in openList or closeList.
        for node in adj[currentNode.node]:
            for x in openList:
                if(x.node == node.v):
                    continue
            
            for x in closeList:
                if(x.node == node.v):
                    continue
            
            openList.append(pathNode(node.v, currentNode))

    return []

# Driver Code
""" Making the following graph
             src = 0
            / | \
           /  |  \
          1   2   3
         /\   |   /\
        /  \  |  /  \
        4   5 6 7    8
               /
              /
            dest = 9
"""
# The total number of vertices.
V = 10
## Initializing the adjacency list
for i in range(V):
    adj.append([])

addEdge(0, 1, 2)
addEdge(0, 2, 1)
addEdge(0, 3, 10)
addEdge(1, 4, 3)
addEdge(1, 5, 2)
addEdge(2, 6, 9)
addEdge(3, 7, 5)
addEdge(3, 8, 2)
addEdge(7, 9, 5)

# Defining the heuristic values for each node.
h = [20, 22, 21, 10, 25, 24, 30, 5, 12, 0]
path = GBFS(h, V, 0, 9)
for i in range(len(path) - 1):
    print(path[i], end = " -> ")

print(path[(len(path)-1)])


#Block world
a=["B","C","D","A"]
k=['A','B','C','D']
print('THE INITIAL STATE IS:',a)
print('THE FINAL STATE IS:',k)
b=[]
c=[]
d=[]
while True:
    add=str(input("WHICH BLOCK DO YOU WANT TP PICK UP:"))
    if(add=='A'):
            print('A IS PICKED UP AND KEPT ON GROUND')
            b.append(add)
            a.remove(add)
            print("a=",a,"\n","b=",b)
            add=input("WHICH BLOCK DO YOU WANT TP PICK UP:")
            if(add=='D'):
                    print('D IS PICKED UP AND KEPT ON GROUND')
                    c.append(add)
                    a.remove(add)
                    print("a=",a,'\n',"b=",b,'\n',"c=",c)
                    add=input("WHICH BLOCK DO YOU WANT TP PICK UP:")
                    if(add=='C'):
                        print('C IS PICKED UP AND KEPT ON GROUND')
                        d.append(add)
                        a.remove(add)
                        print("a=",a,'\n',"b=",b,'\n',"c=",c,'\n',"d=",d)
                        add=input("WHICH BLOCK DO YOU WANT TP PICK UP:")
                        if(add=='B'):
                                print('B IS PICKED UP AND PLACED ON A')
                                b.append(add)
                                a.remove(add)
                                print("a=",a,'\n',"b=",b,'\n',"c=",c,'\n',"d=",d)
                                add=input("WHICH BLOCK DO YOU WANT TP PICK UP:")
                                if add=='C':
                                    print('C IS PICKED UP AND PLACED ON B')
                                    b.append(add)
                                    d.remove(add)
                                    print("a=",a,'\n',"b=",b,'\n',"c=",c,'\n',"d=",d)
                                    add=input("WHICH BLOCK DO YOU WANT TP PICK UP:")
                                    if add=='D':
                                        print('D IS PICKED UP AND PLACED ON C')
                                        b.append(add)
                                        c.remove(add)
                                        if k==b:
                                            print("a=",a,'\n',"b=",b,'\n',"c=",c,'\n',"d=",d)
                                            print('GOAL STATE HAS BEEN ACHIEVED')
                                            break
                                    elif add=='A'or'B'or'C':
                                        print('ALREADY PICKED UP \n START OVER')
                                        break
                                    else:
                                        print('WRONG INPUT IS GIVEN \n START OVER')
                                elif add=='A'or'B':
                                    print('ALREADY PICKED UP \n START OVER')
                                    break
                                elif add=='D':
                                    print('BLOCKS SHOULD BE PICKED UP IN ORDER \n START OVER')
                                    break
                                else:
                                    print('WRONG INPUT IS GIVEN \n START OVER')
                        elif add=='C' or 'D':
                                print('BLOCKS SHOULD BE PICKED UP IN ORDER \n PICK B NEXT TIME')
                                break
                        elif add=='A':
                                print('TO ACHIEVE GOAL STATE DONT PICK UP "A" PICK THE BLOCKS IN ORDER')
                                break
                        else:
                            print('WRONG INPUT IS GIVEN \n START OVER')
                            break
                    elif add=='A'or'D':
                        print('ALREDY PICKED UP \n START OVER')
                        break
                    elif add=='B':
                        print('BLOCKS SHOULD BE PICKED UP IN ORDER \n START OVER')
                        break
                    else:
                        print('WRONG INPUT IS GIVEN \n START OVER')
                        break
            elif add=='B'or'C':
                print('BLOCKS SHOULD BE PICKED UP IN ORDER \n START OVER')
                break
            elif add=='A':
                print('A ALREADY PICKED UP \n START OVER')
                break
            else:
                print('WRONG INPUT IS GIVEN \n START OVER')
                break
    elif add=='B'or'C'or'D':
        print('BLOCKS SHOULD BE PICKED UP IN ORDER \n START OVER')
        break
    else:
        print('WRONG INPUT IS GIVEN \n START OVER')
        Break

#Puzzle
class Node:
    def _init_(self,data,level,fval):
        self.data = data
        self.level = level
        self.fval = fval

    def generate_child(self):
        x,y = self.find(self.data,'_')
        val_list = [[x,y-1],[x,y+1],[x-1,y],[x+1,y]]
        children = []
        for i in val_list:
            child = self.shuffle(self.data,x,y,i[0],i[1])
            if child is not None:
                child_node = Node(child,self.level+1,0)
                children.append(child_node)
        return children
       
    def shuffle(self,puz,x1,y1,x2,y2):
        if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):
            temp_puz = []
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None
           

    def copy(self,root):
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp    
           
    def find(self,puz,x):
        for i in range(0,len(self.data)):
            for j in range(0,len(self.data)):
                if puz[i][j] == x:
                    return i,j


class Puzzle:
    def _init_(self,size):
        self.n = size
        self.open = []
        self.closed = []

    def accept(self):
        puz = []
        for i in range(0,self.n):
            temp = input().split(" ")
            puz.append(temp)
        return puz

    def f(self,start,goal):
        return self.h(start.data,goal)+start.level

    def h(self,start,goal):
        temp = 0
        for i in range(0,self.n):
            for j in range(0,self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp
       

    def process(self):
        print("Enter the start state matrix \n")
        start = self.accept()
        print("Enter the goal state matrix \n")        
        goal = self.accept()

        start = Node(start,0,0)
        start.fval = self.f(start,goal)
        self.open.append(start)
        print("\n\n")
        while True:
            cur = self.open[0]
            print("")
            print("  | ")
            print("  | ")
            print(" \\\'/ \n")
            for i in cur.data:
                for j in i:
                    print(j,end=" ")
                print("")
            if(self.h(cur.data,goal) == 0):
                break
            for i in cur.generate_child():
                i.fval = self.f(i,goal)
                self.open.append(i)
            self.closed.append(cur)
            del self.open[0]
            self.open.sort(key = lambda x:x.fval,reverse=False)


puz = Puzzle(3)
puz.process()

#Wumpus

world = [
    ["S"," "," "," "],
    ["W","G","P"," "],
    ["S"," ","B"," "],
    ["A","B","P"," "]
]
class wumps_world:
    def wumpsisalive(x,y):
        present_con = world[x][y]
        if (len(present_con) > 1 ):
            present_con = world[x][y]
            objects = [*present_con]
            if "W" in objects:
                print("Wumps ate Agent Agent died")
                return False
            elif "P" in objects:
                print("Agnet fell into pit Agent died")
                return False
            elif ("B" in objects):
                print("Agent entered into breeze room danger detected pit room nearby")
                return True
            elif ("S" in objects):
                print("Stench detected wumps room nearby,be carefull")
                return True
            elif(" " in objects):
                print("its an empty room,you are safe")
                return True
            elif("G" in objects):
                print("Gold found you are the richest person in the world")
                return False
        else:
            print("safe")
            return True
            
    
               
               
    def move(x,y):
        while True:
            Move = input("Enter the direction agent moves: ").lower()
            if Move == "right":
                inital = world[x][y]
                if len([*inital]) > 1:
                    in1,in2= [*inital]
                    world[x][y] = in2
                    inital = in1
                else:
                    inital = world[x][y]
                    world[x][y] = ""
                y += 1
                temp = world[x][y]
                world[x][y] = inital+temp
                [print(w) for w in world]
                if(ww.wumpsisalive(x,y)):
                    pass
                else:
                    break
   
            elif Move == "left":
                inital = world[x][y]
                if len([*inital]) > 1:
                    in1,in2= [*inital]
                    world[x][y] = in2
                    inital = in1
                else:
                    inital = world[x][y]
                    world[x][y] = ""
                y -= 1
                temp = world[x][y]
                world[x][y] = inital+temp
                [print(w) for w in world]
                if(ww.wumpsisalive(x,y)):
                    pass
                else:
                    break

            elif Move == "up":
                inital = world[x][y]
                if len([*inital]) > 1:
                    in1,in2= [*inital]
                    world[x][y] = in2
                    inital = in1
                else:
                    inital = world[x][y]
                    world[x][y] = ""
                x -= 1
                temp = world[x][y]
                world[x][y] = inital+temp
                [print(w) for w in world]
                if(ww.wumpsisalive(x,y)):
                    pass
                else:
                    break
                    
            elif Move == "down":
                inital = world[x][y]
                if len([*inital]) > 1:
                    in1,in2= [*inital]
                    world[x][y] = in2
                    inital = in1
                else:
                    inital = world[x][y]
                    world[x][y] = ""
                x += 1
                temp = world[x][y]
                world[x][y] = inital+temp
                [print(w) for w in world]
                if(ww.wumpsisalive(x,y)):
                    pass
                else:
                    break
                    
            elif Move == "break":
                break
            else:
                print("Enter a valid input")
                

                
                
ww = wumps_world
x = 3 ; y = 0
ww.move(x,y)

#8 queens

print ("Enter the number of queens")
N = int(input())
board = [[0]*N for _ in range(N)]
def attack(i, j):
    #checking vertically and horizontally
    for k in range(0,N):
        if board[i][k]==1 or board[k][j]==1:
            return True
    #checking diagonally
    for k in range(0,N):
        for l in range(0,N):
            if (k+l==i+j) or (k-l==i-j):
                if board[k][l]==1:
                    return True
    return False
def N_queens(n):
    if n==0:
        return True
    for i in range(0,N):
        for j in range(0,N):
            if (not(attack(i,j))) and (board[i][j]!=1):
                board[i][j] = 1
                if N_queens(n-1)==True:
                    return True
                board[i][j] = 0
    return False
N_queens(N)
for i in board:
    print (i)

    
    
#Crypt arithmetic
import string
import itertools
x=input()
y=input()
xy=input()
z=input()
zx=input()
zz=input()
inListNumsAsStringArray = [ [x, y], 
                            [z, zx] ]
inResultsArray = [ xy,
                zz ]
inPossibleNumsAsStr = '0123456789'
def getNumberFromStringAndMappingInfo(inStr, inDictMapping):
    numAsStr = ''
    for ch in inStr:
        numAsStr = numAsStr + inDictMapping[ch] 
    return int(numAsStr)

def solveCryptarithmeticBruteForce(inListNumsAsString, inResultStr, inPossibleNumsAsStr):
    nonZeroLetters = []
    strFromStrList = ''
    for numStr in inListNumsAsString:
        nonZeroLetters.append(numStr[0])
        strFromStrList = strFromStrList + numStr
    nonZeroLetters.append(inResultStr[0])
    strFromStrList = strFromStrList + inResultStr  
    uniqueStrs = ''.join(set(strFromStrList))
    for tup in itertools.permutations(inPossibleNumsAsStr, len(uniqueStrs)):
        dictCharAndDigit = {}
        for i in range(len(uniqueStrs)):
            dictCharAndDigit[uniqueStrs[i]] = tup[i]            
        nonZeroLetterIsZero = False
        for letter in nonZeroLetters:
            if(dictCharAndDigit[letter] == '0'):
                nonZeroLetterIsZero = True
                break
        if(nonZeroLetterIsZero == True):
            continue
        result = getNumberFromStringAndMappingInfo(inResultStr, dictCharAndDigit)     
        testResult = 0
        for numStr in inListNumsAsString:
            testResult = testResult + getNumberFromStringAndMappingInfo(numStr, dictCharAndDigit)        

        if(testResult == result):
            strToPrint = ''
            for numStr in inListNumsAsString:
                strToPrint = strToPrint + numStr + '(' + str(getNumberFromStringAndMappingInfo(numStr, dictCharAndDigit)) + ')' + ' + '
            strToPrint = strToPrint[:-3]
            strToPrint = strToPrint + ' = ' + inResultStr + '(' + str(result) + ')'
            print(strToPrint)
            break

for i in range(len(inResultsArray)):
    solveCryptarithmeticBruteForce(inListNumsAsStringArray[i], inResultsArray[i], inPossibleNumsAsStr)

#Tic tac toe

import random


class TicTacToe:

    def _init_(self):
        self.board = []

    def create_board(self):
        for i in range(3):
            row = []
            for j in range(3):
                row.append('-')
            self.board.append(row)

    def get_random_first_player(self):
        return random.randint(0, 1)

    def fix_spot(self, row, col, player):
        self.board[row][col] = player

    def is_player_win(self, player):
        win = None

        n = len(self.board)

        # checking rows
        for i in range(n):
            win = True
            for j in range(n):
                if self.board[i][j] != player:
                    win = False
                    break
            if win:
                return win

        # checking columns
        for i in range(n):
            win = True
            for j in range(n):
                if self.board[j][i] != player:
                    win = False
                    break
            if win:
                return win

        # checking diagonals
        win = True
        for i in range(n):
            if self.board[i][i] != player:
                win = False
                break
        if win:
            return win

        win = True
        for i in range(n):
            if self.board[i][n - 1 - i] != player:
                win = False
                break
        if win:
            return win
        return False

        for row in self.board:
            for item in row:
                if item == '-':
                    return False
        return True

    def is_board_filled(self):
        for row in self.board:
            for item in row:
                if item == '-':
                    return False
        return True

    def swap_player_turn(self, player):
        return 'X' if player == 'O' else 'O'

    def show_board(self):
        for row in self.board:
            for item in row:
                print(item, end=" ")
            print()

    def start(self):
        self.create_board()

        player = 'X' if self.get_random_first_player() == 1 else 'O'
        while True:
            print(f"Player {player} turn")

            self.show_board()

            # taking user input
            row, col = list(
                map(int, input("Enter row and column numbers to fix spot: ").split()))
            print()

            # fixing the spot
            self.fix_spot(row - 1, col - 1, player)

            # checking whether current player is won or not
            if self.is_player_win(player):
                print(f"Player {player} wins the game!")
                break

            # checking whether the game is draw or not
            if self.is_board_filled():
                print("Match Draw!")
                break

            # swapping the turn
            player = self.swap_player_turn(player)

        # showing the final view of board
        print()
        self.show_board()


# starting the game
tic_tac_toe = TicTacToe()
tic_tac_toe.start()

#Decision tree

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

data2 =  pd.read_csv("Time.csv")
data = data2.dropna()

X = pd.get_dummies(data.iloc[:,:-1])
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


classifier = DecisionTreeClassifier()
classifier.fit(X_train,Y_train)


fn = list(pd.get_dummies(data.iloc[:,:-1]).columns)

cn =  [data.columns[-1] for x in range(len(data.axes[0]))]


# tree.export_graphviz(classifier,out_file = "representation.dot", 
#                      feature_names = fn,
#                      class_names = cn ,
#                      filled = True,
#                      rounded = True)

tree.plot_tree(classifier, 
                     feature_names = fn,
                     class_names = cn ,
                     filled = True,
                     rounded = True)
plt.show()

# pre = classifier.predict(X_test)
# acc = accuracy_score(pre,Y_test)
# print(pre)
# print(X)
# print(acc)

#Cluster
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from matplotlib import pyplot

X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

model = Birch(threshold=0.01, n_clusters=2)

model.fit(X)

yhat = model.predict(X)

clusters = unique(yhat)

for cluster in clusters:

 row_ix = where(yhat == cluster)

 pyplot.scatter(X[row_ix, 0], X[row_ix, 1])

pyplot.show()

Linear regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
diabetes = datasets.load_diabetes()
# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]
X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes.target,test_size=0.2)
print(len(X_train ))
print(len(y_train ))
print(len(X_test ))
print(len(y_test ))
# Create linear regression object
regr = linear_model.LinearRegression()

regr.fit(X_train, y_train)
# Make predictions using the testing set
y_pred = regr.predict(X_test)  
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
 % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
# Plot outputs
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
