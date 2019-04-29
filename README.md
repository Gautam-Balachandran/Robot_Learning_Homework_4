# Robot_Learning_Homework_4

This project contains the following folders and files:
	1. src folder : Contains all the source code. Also contains a ".csv" file of the Q-Table used in this program.
	2. Project Report : A detailed report of the code and the results obtained.
	
Requirements : 
The program code makes use of the following python packages :
	1. Numpy
	2. Collections
	3. Sckit Learn package for machine learning. This can be installed using the command
		
		pip install -U scikit-learn
		
	   or can be installedin Conda using the command
	   
		conda install scikit-learn
	

Instructions :

	1. In order to run the program, open the main.py file
	2. Run the file. You will be asked for 2 inputs :
		a. The size of the board for which the program has to run. It can be 2 or 3.
		b. Your choice whether you want to use Function Approximation or Q-Table. Enter 1 for Q-Table and 2 for Function Approximation.
	3. The training for the Q-Table and Function Approximation model are done by playing the learning agent against itself.
	4. The testing part of the code makes use of a Random agent that randomly chooses the lines to be input.
	5. A graphical UI for this program has not been implemented.