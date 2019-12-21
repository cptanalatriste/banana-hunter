from collections import deque
import numpy as np

class BananaManager():
    """
    A manager for our banana agent. It coordinates its learning process.
    """
    def __init__(self, environment_params, epsilon_start=1.0, epsilon_end=0.01,
                 epsilon_decay = 0.995):

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.environment_params = environment_params


    def do_reset(self, environment):
        """
        Prepares the environment for another episode. Any environment-specific
        code should be handled here.
        """
        brain_name = self.environment_params['brain_name']
        environment_info = environment.reset(train_mode=True)[brain_name]

        return environment_info.vector_observations[0]

    def do_step(self, environment, action):
        """
        Performs an actions in the provided environment, Any environment-specific
        code should be handled here.
        """
        brain_name = self.environment_params['brain_name']
        environment_info = environment.step(action)[brain_name]

        next_state = environment_info.vector_observations[0]
        reward = environment_info.rewards[0]
        done = environment_info.local_done[0]

        return next_state, reward, done

    def start_training(self, agent, environment, num_episodes=2000,
                       score_window=100, target_score=15.0,
                       network_file='checkpoint.pth'):
        """
        This methods coordinates the agent training progress. It keeps track
        of all the scores to determine when to finish training and produce a plot.
        """

        all_scores = []
        last_scores = deque(maxlen=score_window)

        epsilon = self.epsilon_start

        for episode in range(1, num_episodes + 1):

            state = self.do_reset(environment)
            current_score = 0.0

            done = False

            while not done:
                action = agent.act(state=state, epsilon=epsilon)
                next_state, reward, done = self.do_step(environment, action)

                agent.step(state, action, reward, next_state, done)

                state = next_state
                current_score += reward

            last_scores.append(current_score)
            all_scores.append(current_score)

            epsilon = max(self.epsilon_end, self.epsilon_decay * epsilon)

            average_score = np.mean(last_scores)
            if episode % score_window == 0:
                print("Episode", episode, "Average score over the last", score_window,
                      " episodes: ", average_score)

            if average_score >= target_score:
                print("Environment solved in ", episode + 1, " episodes. ",
                      "Average score: ", average_score)

                agent.save_trained_weights(network_file=network_file)
                break

        return all_scores
