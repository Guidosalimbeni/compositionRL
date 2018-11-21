'''



https://medium.com/acing-ai/how-i-build-an-ai-to-play-dino-run-e37f37bdf153

https://blog.paperspace.com/dino-run/

https://github.com/Paperspace/DinoRunTutorial

https://gist.github.com/cadurosar/bd54c723c1d6335a43c8

http://edersantana.github.io/articles/keras_rl/

https://ai.intel.com/demystifying-deep-reinforcement-learning/   theory!!!!

https://gist.github.com/EderSantana/c7222daa328f0e885093


'''
import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd

from capture import captureScreenUnity

from directkeys import PressKey, ReleaseKey, W, A, S, D, NP_1, NP_2, NP_3, straight, left, right, reverse, select1, select2,select3, select4, J, K, L, I

from logicReward import CompositionAnalysis

import time
# =============================================================================
# class CompositionGame():
#     
#     def __init__(self):
#         self.name = "my composition in Unity3D"
#     
#     
#     def _update_state(self, action):
#         
#         """
#         Input: action and states
#         Ouput: new states and reward
#         """
#         
#         state = self.state
#         
# 
# 
# =============================================================================

class CompGame(object):
    def __init__(self):
        self.reset()
        self.act(4)
        


    def _get_reward(self):
        

        logicReward = CompositionAnalysis(captureScreenUnity())
        maskForeground, self.visualScore, SlopeVisualBalance = logicReward.VisualBalanceForeground()
    #• TODO implenent reward logic
        if self.visualScore > 0.80:
            visualScoreReward = 0.5
# =============================================================================
#         elif visualScore > 0.80:
#             visualScoreReward = - 0.005
# =============================================================================
        else:
            visualScoreReward = -0.01
        
        print (self.visualScore,visualScoreReward)
        return visualScoreReward


    def _is_over(self):
        if self.visualScore > 0.81:
            print ("got 0.85 Visual Score")
            return True
        else:
            return False

    def observe(self):
        
        # get the screen capture
        
        self.screen = captureScreenUnity() # 23 x 15 = 345 (1,345)
        
        # input is (1,345) one row and 100 pixels
        
        return self.screen.reshape((1, -1))

    def act(self, action):
        
        if action == 0:
            straight()
        elif action == 1:
            left()
        elif action == 2:
            right()
        elif action == 3:
            reverse()
        elif action == 4:
            select1()
        elif action == 5:
            select2()
        elif action == 6:
            select3()
        else: 
            select4()
            
        
        reward = self._get_reward()
        game_over = self._is_over()
        return self.observe(), reward, game_over

    def reset(self):
        
        # TODO send a key to reset UNITY to initial position of the 3 objects in the scene
        
        pass

# 3w3ww31ddwww13dwwwww1ww4d2ws31dw4aws24a44a1ws1w33424w1sdwww2211s1aswsss234sw4ws21w4s32ws1ww4dww122a3wwa3w3www34wdwwsw1
###### continue from here...


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1], game_over?]
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[1]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            game_over = self.memory[idx][1]

            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            Q_sa = np.max(model.predict(state_tp1)[0])
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets


if __name__ == "__main__":
    
    time.sleep(5)
    # parameters
    epsilon = .8  # exploration
    num_actions = 8  # [move_left, stay, move_right, move back, select 1, selec2 , select3]
    epoch = 100
    max_memory = 500
    hidden_size = 100
    batch_size = 50
    grid_size = 10

    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(1035,), activation='relu'))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions))
    model.compile(sgd(lr=.2), "mse")

    # If you want to continue training from a previous model, just uncomment the line bellow
    # model.load_weights("model.h5")
    
    # TODO  check that I implemented correctly the game as reading from Unity and dealing with the logic here
    # Define environment/game
    env = CompGame()
    
    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    win_cnt = 0
    for e in range(epoch):
        print ("epoch number starts" + str(e))
        loss = 0.3
        env.reset()
        game_over = False
        # get initial input this is the first screen grab from Unity 3D
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t # 23 x 15 = 345 (1,345)
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])
            
            # TODO here I need to link to the correct action to UNITY3D 
            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if reward == 1:
                #♠ TODO save the image with a good score..
                win_cnt += 1

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            loss += model.train_on_batch(inputs, targets)
            # loss += model.train_on_batch(inputs, targets)[0]
        print("Epoch {:03d}/999 | Loss {:.4f} | Win count {}".format(e, loss, win_cnt))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)