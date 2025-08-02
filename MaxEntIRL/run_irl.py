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
        if filename.endswith(".pkl") and not os.path.isdir(os.path.join(feature_dir, filename)):
            filepath = os.path.join(feature_dir, filename)
            with open(filepath, "rb") as f:
                features.append(pickle.load(f))
    
    return features


def convert_features_to_array(features, feature_names=['velocity', 'a_long', 'jerk_long', 'a_lateral', 'thw_front', 'thw_rear']):
    """
    Convert feature dictionary to numpy array by aggregating time-series features.
    
    Args:
        features: Dictionary with feature names as keys and arrays as values, or numpy array
        feature_names: List of feature names in the desired order
        
    Returns:
        numpy array of aggregated feature values
    """
    if isinstance(features, dict):
        feature_values = []
        
        # Extract and aggregate each feature type (take mean across time steps)
        for feature_name in feature_names:
            feature_array = np.array(features[feature_name])
            if len(feature_array) > 0:
                # Take mean to get single value per feature
                feature_values.append(np.mean(feature_array))
        
        return np.array(feature_values)
    else:
        return np.array(features)

def maxent_irl(features):
    """
    Maximum Entropy Inverse Reinforcement Learning
    
    Args:
        features: List of loaded feature data from pkl files
    """
    # Parameters for IRL    
    feature_num = 6  # Updated to match actual feature count
    n_iters = 200
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lam = 0.01
    lr = 0.05
    
    # Initialize Adam optimizer variables
    pm = None
    pv = None    
    
    training_log = {'iteration': [], 'average_feature_difference': [],
                    'average_log-likelihood': [],
                    'average_human_likeness': [],
                    'theta': []}
    
    # Initialize weights
    theta = np.random.normal(0, 0.05, size=feature_num)
    
    for i in range(n_iters):
        print(f'iteration: {i + 1}/{n_iters}')

        feature_exp = np.zeros([feature_num])
        human_feature_exp = np.zeros([feature_num])
        log_like_list = []
        iteration_human_likeness = []
        num_traj = 0

        # compute on each scene
        for scene_idx, scene_data in enumerate(features):
            # compute on each frame
            for frame_data in scene_data:
                start_frame = frame_data["start_frame"]
                frame_features = frame_data["frame_features"]
                
                agent_rollout_features = frame_features['agent_rollout_features']
                agent_ground_truth_features = frame_features['agent_ground_truth_features']
                               
                # Process each agent
                for agent_id, gt_features in agent_ground_truth_features.items():
                    if agent_id not in agent_rollout_features:
                        continue
                    
                    # Get the rollout features for the current agent
                    rollout_features = agent_rollout_features[agent_id]
                    
                    # Collect trajectories for this agent
                    agent_trajs = []
                    
                    # Process each rollout trajectory
                    for rollout_traj in rollout_features:
                        # Convert feature dict to array
                        rollout_array = convert_features_to_array(rollout_traj)
                        
                        # Compute the reward for this trajectory
                        reward = np.dot(rollout_array, theta)

                        # Store trajectory info: (reward, features)
                        agent_trajs.append((reward, rollout_array))

                    
                    # Convert ground truth features to array
                    gt_array = convert_features_to_array(gt_features)
                    
                    # Add ground truth features as the "human" trajectory
                    human_reward = np.dot(gt_array, theta)
                    agent_trajs.append((human_reward, gt_array))

                    if not agent_trajs:
                        continue
                
                    num_traj += 1 # Count each expert trajectory as one demonstration
                
                # Calculate probability of each trajectory
                rewards = [traj[0] for traj in agent_trajs]
                # Prevent overflow by subtracting max reward
                max_reward = np.max(rewards)
                exp_rewards = np.exp(np.array(rewards) - max_reward)
                probs = exp_rewards / np.sum(exp_rewards)

                # Calculate feature expectation with respect to the policy
                traj_features = np.array([traj[1] for traj in agent_trajs])
                feature_exp += np.dot(probs, traj_features)

                # Calculate likelihood of human trajectory (assuming last one is human)
                log_like = np.log(probs[-1] + eps)  # Add small epsilon to prevent log(0)
                log_like_list.append(log_like)

                # Calculate human likeness for top trajectories
                top_k = min(3, len(probs))
                top_indices = np.argsort(probs)[-top_k:][::-1]
                top_human_likeness = [np.linalg.norm(agent_trajs[idx][1] - gt_array) for idx in top_indices]
                iteration_human_likeness.extend(top_human_likeness)

                # Add human trajectory features to expectation
                human_feature_exp += gt_array

        if num_traj == 0:
            print("No trajectories found in this iteration")
            continue

        # Compute gradient
        grad = human_feature_exp / num_traj - feature_exp / num_traj - 2 * lam * theta
        grad = np.array(grad, dtype=float)

        # Update weights using Adam optimization
        if pm is None:
            pm = np.zeros_like(grad)
            pv = np.zeros_like(grad)

        pm = beta1 * pm + (1 - beta1) * grad
        pv = beta2 * pv + (1 - beta2) * (grad * grad)
        mhat = pm / (1 - beta1 ** (i + 1))
        vhat = pv / (1 - beta2 ** (i + 1))
        # Update theta with learning rate and Adam update
        update_vec = mhat / (np.sqrt(vhat) + eps)
        theta += lr * update_vec

        # Record info during training
        training_log['iteration'].append(i + 1)
        training_log['average_feature_difference'].append(
            np.linalg.norm(human_feature_exp / num_traj - feature_exp / num_traj))
        training_log['average_log-likelihood'].append(np.mean(log_like_list))
        training_log['average_human_likeness'].append(np.mean(iteration_human_likeness))
        training_log['theta'].append(theta.copy())
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}: Log-likelihood = {np.mean(log_like_list):.4f}")
    
    return theta, training_log

if __name__ == "__main__":
    # Load features from the specified directory
    features = load_features()
    
    # Run the IRL process
    learned_theta, log = maxent_irl(features=features)
    
    print(f"Final learned weights: {learned_theta}")
    
    # Save the results
    with open("irl_results.pkl", "wb") as f:
        pickle.dump({"theta": learned_theta, "training_log": log}, f)