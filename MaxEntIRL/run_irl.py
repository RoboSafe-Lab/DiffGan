import numpy as np
import os
import pickle

def load_features():
    """Load features from the specified directory."""
    feature_dir = "irl_features_output"
    if not os.path.exists(feature_dir):
        raise FileNotFoundError(f"Feature directory {feature_dir} does not exist.")
    
    features = []
    for filename in os.listdir(feature_dir):
        if filename.endswith(".pkl"):
            filepath = os.path.join(feature_dir, filename)
            with open(filepath, "rb") as f:
                features.append(pickle.load(f))
    
    return features

def maxent_irl(features):
    # parameters for IRL    
    feature_num = 10
    n_iters = 200
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lam = 0.01
    lr = 0.05
    
    
    training_log = {'iteration': [], 'average_feature_difference': [],
                    'average_log-likelihood': [],
                    'average_human_likeness': [],
                    'theta': []}
    
    """training the weights under each iteration"""
    human_features = features[0]
    buffer_scenes = features[1]

    theta = np.random.normal(0, 0.05, size=feature_num)
    for i in range(n_iters):
        logger.info(f'iteration: {i + 1}/{n_iters}')

        feature_exp = np.zeros([feature_num])
        human_feature_exp = np.zeros([feature_num])
        index = 0
        log_like_list = []
        iteration_human_likeness = []
        num_traj = 0

        for scene in buffer_scenes:
            # compute on each scene
            scene_trajs = []
            for trajectory in scene:
                reward = np.dot(trajectory[2], theta)
                scene_trajs.append((reward, trajectory[2], trajectory[3]))  # reward, feature vector, human likeness

            # calculate probability of each trajectory
            rewards = [traj[0] for traj in scene_trajs]
            # data overflow method
            max_reward = np.max(rewards)
            probs = [np.exp(reward-max_reward) for reward in rewards]
            probs = probs / np.sum(probs)

            # calculate feature expectation with respect to the weights
            traj_features = np.array([traj[1] for traj in scene_trajs])
            feature_exp += np.dot(probs, traj_features)  # feature expectation

            # calculate likelihood
            log_like = np.log(probs[-1] / np.sum(probs))
            log_like_list.append(log_like)

            # select trajectories to calculate human likeness
            # extracting the indices of the top 3 highest values in probs
            idx = probs.argsort()[-3:][::-1]
            iteration_human_likeness.append(np.min([scene_trajs[i][-1] for i in idx]))

            # calculate human trajectory feature
            human_feature_exp += human_features[index]

            # go to next trajectory
            num_traj += 1
            index += 1

        # compute gradient
        grad = human_feature_exp - feature_exp - 2 * lam * theta
        grad = np.array(grad, dtype=float)

        # update weights using Adam optimization
        if pm is None:
            pm = np.zeros_like(grad)
            pv = np.zeros_like(grad)

        pm = beta1 * pm + (1 - beta1) * grad
        pv = beta2 * pv + (1 - beta2) * (grad * grad)
        mhat = pm / (1 - beta1 ** (i + 1))
        vhat = pv / (1 - beta2 ** (i + 1))
        update_vec = mhat / (np.sqrt(vhat) + eps)
        theta += lr * update_vec

        # record info during the training
        training_log['iteration'].append(i + 1)
        training_log['average_feature_difference'].append(
            np.linalg.norm(human_feature_exp / num_traj - feature_exp / num_traj))
        training_log['average_log-likelihood'].append(np.sum(log_like_list) / num_traj)
        training_log['average_human_likeness'].append(np.mean(iteration_human_likeness))
        training_log['theta'].append(theta.copy())

if __name__ == "__main__":
    # Load features from the specified directory
    features = load_features()
    
    import pdb; pdb.set_trace()
    # Run the IRL process
    maxent_irl(features=features)

    
    # Save the results or perform further analysis as needed