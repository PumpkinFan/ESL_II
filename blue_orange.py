# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.inspection import DecisionBoundaryDisplay
# from scipy import stats



def generate_observations(base_mean, n_observations):
    # base mean is equivalent to (1,0)^T for BLUE 
    # and (0,1)^T for ORANGE in text
    base_mean = np.array(base_mean)
    cov1 = np.identity(len(base_mean))
    m_ks = np.random.multivariate_normal(base_mean, cov1, size=10)
    
    observations = []
    cov2 = cov1 / 5
    
    print(f"m_ks = {m_ks}")
    for _ in range(n_observations):
        k_sel = np.random.choice(m_ks.shape[0], 1)
        
        m_k_sel = np.ravel(m_ks[k_sel])
        
        obsv = np.ravel(np.random.multivariate_normal(m_k_sel, cov2, size=1))
        observations.append(obsv)
    return np.array(observations)

if __name__ == "__main__":
    fig, ax = plt.subplots(1,1)
    
    N_OBS = 100
    blue_obs = generate_observations((1,0), N_OBS)
    orange_obs = generate_observations((0,1), N_OBS)
        
    kneighbors_clf = neighbors.KNeighborsClassifier(n_neighbors=15, weights='uniform')
    full_dataset = np.vstack((blue_obs, orange_obs))
    full_targets = np.hstack((np.ones(N_OBS), np.zeros(N_OBS)))
    
    kneighbors_clf.fit(full_dataset, full_targets)
    
    custom_cmap = matplotlib.colors.ListedColormap(["bisque", "lightskyblue"])
    DecisionBoundaryDisplay.from_estimator(kneighbors_clf, 
                                           full_dataset, 
                                           ax=ax, 
                                           cmap=custom_cmap)

    ax.plot(blue_obs[:, 0], blue_obs[:, 1], "o", c="blue")
    ax.plot(orange_obs[:, 0], orange_obs[:, 1], "o", c="orange")
        
    