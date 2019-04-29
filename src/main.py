# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:41:57 2019

@author: Gautam Balachandran
"""

import numpy as np
from DotsAndBoxes import *

def training(num_games,q_table,grid_size,choice):
    random_training_rate = 0.2*num_games
    for loop in range(num_games):
        if loop%random_training_rate == 0:
            d = DotsandBoxes(size = grid_size,rand = True)
        else:
            d = DotsandBoxes(size = grid_size)

        if choice == 1:
            d.use_q_table = True
        else:
            d.use_q_table = False
        d.current_player = "Learning Agent"
        while d.empty_boxes:
            if d.use_q_table:
                q_table = d.play(q_table)
            else:
                d.playWithNN()
    return q_table

def testing(q_table, num_games,grid_size,choice):
    wins = 0
    losses = 0
    draws = 0
    for _ in range(num_games):
        d = DotsandBoxes(grid_size)
        if choice == 1:
            d.use_q_table = True
        else:
            d.use_q_table = False
        i = 0
        while d.empty_boxes:
            if i%2 == 0:
                d.current_player = "Random Agent"
            else:
                d.current_player = "Learning Agent"
            if d.use_q_table:
                q_table = d.play(q_table)
            else:
                d.playWithNN()
            i += 1

        if d.wins == 1:
            wins += 1
        elif d.losses == 1:
            losses += 1
        else:
            draws += 1

    return wins,draws,losses







if __name__== "__main__":
#    num_games_training = int(input("Enter the training iterations : "))
    num_games_training = int(input("Enter the number of training iterations : "))
    num_games_testing = 100
#    Q_Table_file = open("Q_TABLE.txt","a")
#    Q_Table_file.truncate(0) # Clears file
    grid_size = int(input("Enter the board size : "))
    choice = int(input("Enter 1 for Q-Table and 2 for Function Approximation : "))
    q_table = None
    q_table = training(num_games_training,q_table,grid_size,choice)
    String = "Q-Table for "+str(grid_size)+"X"+str(grid_size)+" Game :\n"
#    Q_Table_file.write(String)
#    np.savetxt("Q-Table.csv",q_table,delimiter=',')
#    Q_Table_file.write(str(q_table))
    print("--------------------------TESTING----------------------------------")
    wins,draws,losses = testing(q_table,num_games_testing,grid_size,choice)
    print("Win percentage for " + str(num_games_training)+" iterations testing :",(wins/num_games_testing)*100)
    print("Loss percentage : ",(losses/num_games_testing)*100)
    print("Draws percentage : ",(draws/num_games_testing)*100)