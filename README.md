# sarsa-mountaincar
An implementation of the SARSA algorithm in the Mointain Car domian.

This project implements SARSA(lambda) on the Mointain Car domain with a Boltzmann control policy. For further reference see the book referenced below.

@book{Sutton98,
	Address = {Cambridge, MA},
	Author = {Richard S. Sutton and Andrew G. Barto},
	Date-Added = {2015-12-07 13:46:04 +0000},
	Date-Modified = {2015-12-07 13:47:07 +0000},
	Publisher = {MIT Press},
	Title = {Reinforcement Learning: An Introduction},
	Year = {1998}}

To run the demo experiment launch the mountaincar.py script with

python mountaincar.py

To see what parameters can be set to run the simulation run

python mountaincar.py --help
