# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 15:01:46 2019

@author: Gautam Balachandran
"""
import numpy as np
import sys
import math
from collections import defaultdict as dd
np.set_printoptions(threshold=sys.maxsize)
import DotsAndBoxes_NN as dbn

class DotsandBoxes():

    def __init__(self,size=2,rand = False):
        self.size = size
        self.num_rows = size
        self.num_cols = 2*(size+1)
        self.num_lines = self.num_cols*self.num_rows # Number of possible lines
        self.state_size = int(math.pow(2,self.num_lines)) # Number of unique states
#        self.q_table = np.zeros((self.state_size, self.num_lines))
        self.lines_list = np.zeros((self.num_lines))
        self.reward = {"Learning Agent":0,"Random Agent":0}
        self.current_player = None
        self.current_state = 0
        self.update_state = 0
        self.score = dd(int)
        self.boxes = np.zeros((size*size)) # represents already completed boxes
        self.vertical = np.zeros((size*(size+1)))
        self.horizontal = np.zeros((size*(size+1)))
        self.wins = 0
        self.losses = 0
        self.box_indices = []
        self.empty_boxes = True # Flag that tells if there are still empty boxes left
        self.random = rand
        self.use_q_table = True

        for i in range(int(math.pow(size,2))):
            j = i+1
            box = [j,j+self.num_rows,j+self.num_cols+int(i/self.num_rows),j+self.num_cols+int(i/self.num_rows)+1]
            self.box_indices.append(box)

        # Rates
        self.epsilon = 0
        self.learning_rate = 0.8
        self.discount_rate = 0.8

    def update_QTable(self,reward_current,move,q_table):
        Q_current = q_table[self.current_state ,move]
#        if self.current_state < self.q_table.shape[0]-1:
#            next_state = self.current_state+1
#        else:
#            next_state = self.current_state
        inner_inner = reward_current + self.discount_rate*(np.max(q_table[self.update_state ,:]))
        inner = inner_inner -Q_current
        Q_current += self.learning_rate*(np.abs(inner))
        q_table[self.current_state ,move] = Q_current
        return q_table

    def play(self,q_table):
        if q_table is None:
            q_table = np.zeros((self.state_size, self.num_lines))
        move = 0
        if (self.boxes == 1).sum() == self.size*self.size: # Checks if any empty boxes left
            print("All Boxes filled")
            print("-------------------------------------//////-------------------------------------------------")
            self.empty_boxes = False
            if self.reward["Learning Agent"] > self.reward["Random Agent"]:
                self.wins = 1
                q_table = self.update_QTable(5,move,q_table)
            elif self.reward["Learning Agent"] < self.reward["Random Agent"]:
                self.losses = 1
#            else:
#                if self.current_player == "Learning Agent":
#                    self.losses = 1
#                else:
#                    self.wins = 1
            return

        self.current_state = self.update_state
        if self.current_player == "Learning Agent":
            print("CURRENT STATE : ",self.current_state)
            if self.random:
                move = np.random.randint(0,self.num_lines)
            else:
                values = q_table[self.current_state,:]
                print("Current State Q_Values : ",values)
                length = len(values)
                values.sort()
                print("Sorted Values : ",values)
                l = 1
                max_val = q_table[self.current_state,length-l]
                move = np.where(q_table[self.current_state,:] == max_val)
                print("Move Values: ",move[0])
                pos = len(move[0])
                if pos>1:
                    m = np.random.randint(0,pos)
                else:
                    m = 0
                move = (move[0])[m]
                print("INNER MOVE : ",move+1)
        else:
            move = np.random.randint(0,self.num_lines)
        print("Lines List : ",self.lines_list)
        while self.lines_list[move] == 1:
            if self.random:
                move = np.random.randint(0,self.num_lines)
            else:
                if self.current_player == "Learning Agent"and l<length:
                    l += 1
                    max_val = q_table[self.current_state,length-l]
                    move = np.where(q_table[self.current_state,:] == max_val)
                    pos = len(move[0])
                    if pos>0:
                        m = np.random.randint(0,pos)
                    else:
                        m = 0
                    move = (move[0])[m]
                else:
                    move = np.random.randint(0,self.num_lines)

#        self.move_sequence.append(move)

        print("MOVE : ",move+1)
        print("Q_Value : ",q_table[self.current_state ,move])
        self.lines_list[move] = 1

        string = ''.join([str(int(l)) for l in self.lines_list])
        self.update_state = int(string,2) # Binary string to decimal

        val = (self.lines_list == 1).sum()
        if val>=4:
            for i in range(self.size*self.size):
                count = 0
                for j in range(len(self.box_indices[i])):
                    p = self.box_indices[i][j]
                    if self.lines_list[p-1] == 1:
                        count += 1
                        if count == 4:
                            if self.boxes[i] != 1:
                                reward_current = self.reward[self.current_player]
                                self.reward[self.current_player]  = reward_current+1
                                self.boxes[i] = 1
                                if self.current_player == "Learning Agent":
                                    q_table = self.update_QTable(reward_current,move,q_table)

        print("Boxes : ",self.boxes)
        print("Lines List : ",self.lines_list)
        print("Updated State : ",self.update_state)
        print("Current Player : ",self.current_player)
        print("Number of lines : ",val)
        print("REWARDS : ",self.reward)
        print("==================================")

        return q_table

    def make_epsilon_greedy_policy(estimator, epsilon, nA,observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    def playWithNN(self):
        move = 0
        self.current_state = self.update_state
        estimator = dbn.Estimator(self.lines_list,self.current_state)
        if (self.boxes == 1).sum() == self.size*self.size: # Checks if any empty boxes left
            print("All Boxes filled")
            print("-------------------------------------//////-------------------------------------------------")
            self.empty_boxes = False
            if self.reward["Learning Agent"] > self.reward["Random Agent"]:
                self.wins = 1
                q_values_next = estimator.predict(self.current_state)
                reward = 5
                td_target = reward + self.discount_rate * np.max(q_values_next)
                estimator.update(self.update_state, move, td_target)
            elif self.reward["Learning Agent"] < self.reward["Random Agent"]:
                self.losses = 1

        move_list = self.make_epsilon_greedy_policy(estimator,self.epsilon, self.num_lines,self.current_state)
        move = np.random.choice(np.arange(len(move_list)), p=move_list)
        while self.lines_list[move]==1:
            move = np.random.choice(np.arange(len(move_list)), p=move_list)

        print("MOVE : ",move+1)
        self.lines_list[move] = 1

        string = ''.join([str(int(l)) for l in self.lines_list])
        self.update_state = int(string,2) # Binary string to decimal

        q_values_next = estimator.predict(self.update_state)
        td_target = self.reward[self.current_player] + self.discount_rate * np.max(q_values_next)
        estimator.update(self.update_state, move, td_target)





