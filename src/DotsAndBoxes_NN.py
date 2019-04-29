# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:33:23 2019

@author: Gautam Balachandran
"""

import numpy as np
import sys
import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
  sys.path.append("../")

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler



class Estimator():
    def __init__(self,action_space,current_state,state_size):
        self.models = []
        self.state_size = state_size
        actions_size = len(action_space)
        for _ in range(actions_size):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(current_state)], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        observation_samples = np.arange(0,self.state_size,1)
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(observation_samples)
        scaled = scaler.transform([state])
        featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        featurized = featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """ Predicts the action. """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """ Updates the estimator parameters for a given state and action towards the target y. """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])
