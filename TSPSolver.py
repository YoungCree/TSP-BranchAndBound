#!/usr/bin/python3
from copy import deepcopy

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

class State:
	def __init__(self, lower_bound, matrix, partial_path):
		self.lower_bound = lower_bound
		self.matrix = matrix
		self.partial_path = partial_path

	def path_len(self):
		return len(self.partial_path)

	def get_city(self):
		return self.partial_path[self.path_len()-1]

	def __lt__(self, other):
		# We like deeper paths, so return the deeper (if at least 2 levels deeper) city even if it has a worse cost
		if self.path_len() > other.path_len() + 2:
			return True
		if other.path_len() > self.path_len() + 2:
			return False
		return self.lower_bound < other.lower_bound



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy( self,time_allowance=60.0 ):
		pass
	
	
	
	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
		
	def branchAndBound(self, time_allowance=60.0):
		bssf = self.defaultRandomTour(time_allowance)['soln']
		best_cost = bssf.cost
		cities = self._scenario.getCities()
		results = {}

		num_cities = len(cities)
		heap = []
		num_states_gen = 1
		num_states_pruned = 0
		num_solutions = 0
		num_bffs_updates = 0
		max_q_size = 0

		# Grab our initial state starting at city 0 and push it onto the queue
		initial_matrix, initial_lower_bound = self.create_initial_matrix(cities)
		initial_state = State(initial_lower_bound, initial_matrix, [cities[0]])
		heapq.heappush(heap, initial_state)

		# Start the timer
		initial_time = time.time()

		# Run for the allotted time or until nothing is left in the queue (optimal solution)
		while time.time() - initial_time < time_allowance and len(heap) > 0:
			# Grab the best state off the queue
			curr_state = heapq.heappop(heap)

			if curr_state.lower_bound < best_cost:
				for city in cities:
					# We don't want to look at cities we've already visited
					if city not in curr_state.partial_path:
						# Create the new state
						new_state = self.create_new_state(city, curr_state)
						num_states_gen += 1

						# If we've reached a new solution
						if new_state.path_len() == num_cities:
							bssf = TSPSolution(new_state.partial_path)
							num_solutions += 1
							# If we've reached a better solution, update the best cost
							if bssf.cost < best_cost:
								best_cost = bssf.cost
								num_bffs_updates += 1

						# We still have cities to visit
						elif new_state.lower_bound < best_cost:
							heapq.heappush(heap, new_state)

						# Worthless state, prune it
						else:
							num_states_pruned += 1

			# Worthless state, prune it
			else:
				num_states_pruned += 1
				num_states_gen += 1

			# Keep track of max queue size
			max_q_size = max(len(heap), max_q_size)

		# Store results
		end_time = time.time()
		results['cost'] = best_cost
		results['time'] = end_time - initial_time
		results['count'] = num_solutions
		results['soln'] = bssf
		results['max'] = max_q_size
		results['total'] = num_states_gen
		results['pruned'] = num_states_pruned
		return results



	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy( self,time_allowance=60.0 ):
		pass

	# Creates the initial matrix
	def create_initial_matrix(self, cities):
		# Fill a matrix with np.inf and then add values
		matrix = np.full((len(cities), len(cities)), np.inf)
		for f_index, f_city in enumerate(cities):
			for t_index, t_city in enumerate(cities):
				if f_index == t_index:
					continue
				matrix[f_index][t_index] = f_city.costTo(t_city)

		# Reduce our matrix
		lower_bound = 0
		for row in range(matrix.shape[0]):
			row_min = np.min(matrix[row])
			matrix[row] = matrix[row] - row_min
			lower_bound += row_min

		for col in range(matrix.shape[1]):
			col_min = np.min(matrix[:, col])
			matrix[:, col] = matrix[:, col] - col_min
			lower_bound += col_min

		return matrix, lower_bound

	def create_new_state(self, city, curr_state):
		matrix = curr_state.matrix.copy()

		# infinity out the row, column, and index associated with the two cities
		inf_row = curr_state.get_city()._index
		matrix[inf_row] = np.inf

		inf_col = city._index
		matrix[:, inf_col] = np.inf

		matrix[inf_col][inf_row] = np.inf

		# Reduce our matrix
		lower_bound = curr_state.lower_bound.copy() + curr_state.matrix[inf_row][inf_col].copy()
		for row in range(matrix.shape[0]):
			row_min = np.min(matrix[row])

			if np.isinf(row_min):
				continue

			matrix[row] = matrix[row] - row_min
			lower_bound += row_min

		for col in range(matrix.shape[1]):
			col_min = np.min(matrix[:, col])

			if np.isinf(col_min):
				continue

			matrix[:, col] = matrix[:, col] - col_min
			lower_bound += col_min

		# Add the new city to the partial_path, create the new state
		partial_path = curr_state.partial_path.copy() + [city]

		return State(lower_bound, matrix, partial_path)

