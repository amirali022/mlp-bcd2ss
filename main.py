### BCD to Seven Segment Using MLP

#### Setup

import numpy as np
from sklearn.neural_network import MLPClassifier

#### Generate Input And Output

#	l: low
#	h: high
def BCD( l = 0, h = 1):
	return np.array( [
		# A  B  C  D
		[ l, l, l, l],	# 0
		[ l, l, l, h],	# 1
		[ l, l, h, l],	# 2
		[ l, l, h, h],	# 3
		[ l, h, l, l],	# 4
		[ l, h, l, h],	# 5
		[ l, h, h, l],	# 6
		[ l, h, h, h],	# 7
		[ h, l, l, l],	# 8
		[ h, l, l, h]	# 9
	])

X_train = BCD()

T = np.array( [
	# a  b  c  d  e  f  g
	[ 1, 1, 1, 1, 1, 1, 0],	# 0
	[ 0, 1, 1, 0, 0, 0, 0],	# 1 
	[ 1, 1, 0, 1, 1, 0, 1],	# 2
	[ 1, 1, 1, 1, 0, 0, 1],	# 3
	[ 0, 1, 1, 0, 0, 1, 1],	# 4
	[ 1, 0, 1, 1, 0, 1, 1],	# 5
	[ 1, 0, 1, 1, 1, 1, 1],	# 6
	[ 1, 1, 1, 0, 0, 0, 0],	# 7
	[ 1, 1, 1, 1, 1, 1, 1],	# 8
	[ 1, 1, 1, 1, 0, 1, 1]	# 9
])

#### Create and Train Model

bcd2SS = MLPClassifier(
	hidden_layer_sizes=( 4),
	activation="logistic",
	solver="sgd",
	learning_rate="constant",
	learning_rate_init=1,
	max_iter=500,
	shuffle=True,
	tol=1e-4
)

bcd2SS.fit( X_train, T)

print( "Train score: ", bcd2SS.score( X_train, T))

X_test = BCD( 0.15, 0.85)

print( f"Test Set:\n{ X_test}")
print( f"Network Prediction:\n{ bcd2SS.predict( X_test)}")
print( f"Test score: { bcd2SS.score( X_test, T)}")