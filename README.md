# BPP_interview_Adam_Fajkus

BLUE PRINT POWER

DS Interview Project
Interview Project - Optimisation Engineer
2023

Adam Fajkus

How to run:
- You should open the script named run.py and then simply run the code for the specified date
- The date is prespecified to run the date inside the assignment
- I used these packages to run the optimization:
python 			3.8.2
	numpy              	1.22.3
	pandas             	1.4.1 
	Pyomo              	6.6.1
GLKP			4.65

My comments:
- I am not used to work with pyomo, because my current setup is AMPL with the Gurobi
- but I wanted to use an open-source package and try to learn something new
- At first I tried to start with maximum allowed time step; the results were good enough
- After taking a closer look at the data, I saw that the timestamp changes and I hope it is not an error in the data
- I realized that there are some good opportunities to look at (in the smaller steps)
- So I decided to optimize the case for the smallest interval possible.
- The pros were that the performance was good and the revenues was also very good
- Results also passed my own validation test.

Bonus points:
Utilize industry best practices for software development
- I think I have met the industry standards for software development
Suggestions on additional inputs or information that could improve the optimization
- In this case we have optimized the Perfect foresight case
- As in the real life we do not have perfect predictions, my suggestion would be to optimize the battery against prediction or multiple predictions
- We can also deal with the VaR to not be that aggresive
- In real life and some markets you you are also dealing with the ancillary services, imbalance, intraday trading, and outages which add complexity to the model
- I don't have in-depth knowledge about US markets, but my feeling is that you also should look at Marginal Cost Losses ($/MWHr) and Marginal Cost Congestion.
