import AI_Proj3
import torch
from torch import nn
from matplotlib import pylab as plt
from time import time
import numpy as np
import random

MAX_TEST = 100
MAX_TRAIN = 1000

FULL_GRID_STATE = AI_Proj3.GRID_SIZE**2
H_LAYER_1 = 100
H_LAYER_2 = 50
H_LAYER_3 = 25
BOT_ACTIONS = 9
GAAMA = 0.9

ACTIONS_ID = {
"IDLE" : int(0),
"NORTH" : int(1),
"SOUTH" : int(5),
"EAST" : int(3),
"WEST" : int(7),
"NORTH_EAST" : int(2),
"NORTH_WEST" : int(8),
"SOUTH_EAST" : int(4),
"SOUTH_WEST" : int(6)
}

class QModel(nn.Module):
  def __init__(self):
    super(QModel, self).__init__()
    # self.input_layer = torch.nn.Linear( in_features = FULL_GRID_STATE, out_features = H_LAYER_2, bias=True )
    self.input_layer = torch.nn.Linear( in_features = FULL_GRID_STATE, out_features = H_LAYER_1, bias=True )
    self.layer_1 = torch.nn.Linear( in_features = H_LAYER_1, out_features = H_LAYER_2, bias = True )
    self.layer_2 = torch.nn.Linear( in_features = H_LAYER_2, out_features = H_LAYER_3, bias = True)
    self.layer_3 = torch.nn.Linear( in_features = H_LAYER_3, out_features = BOT_ACTIONS, bias = True)

  def forward(self, input_tensor):
    output = self.input_layer(input_tensor)
    output = nn.ReLU()(output)
    output = self.layer_1(output)
    output = nn.ReLU()(output)
    output = self.layer_2(output)
    output = nn.ReLU()(output)
    output = self.layer_3(output)
    return output

class LEARN_CONFIG(AI_Proj3.SHIP):
    def __init__(self):
        super(LEARN_CONFIG, self).__init__()
        self.q_model = QModel()
        self.optimizer = torch.optim.ASGD(self.q_model.parameters(), lr=AI_Proj3.CONVERGENCE_LIMIT)
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.L1Loss()
        # self.loss_fn = torch.nn.CrossEntropyLoss()
        self.losses = []

    def print_losses(self):
        plt.figure(figsize=(10,7))
        plt.plot(self.losses)
        plt.xlabel("Epochs",fontsize=22)
        plt.ylabel("Loss",fontsize=22)
        plt.show()
        self.losses.clear()

    def get_action(self, bot_cell, bot_move, is_parse = False):
        val = -1
        if bot_cell and bot_move:
            delta_x = bot_move[0] - bot_cell[0]
            delta_y = bot_move[1] - bot_cell[1]

            if delta_x == 0 and delta_y == 0:
                val = ACTIONS_ID["IDLE"]
            elif delta_x == 0:
                val = ACTIONS_ID["NORTH"] if delta_y > 0 else ACTIONS_ID["SOUTH"]
            elif delta_y == 0:
                val = ACTIONS_ID["EAST"] if delta_x > 0 else ACTIONS_ID["WEST"]
            elif delta_x > 0:
                val = ACTIONS_ID["NORTH_EAST"] if delta_y > 0 else ACTIONS_ID["SOUTH_EAST"]
            else:
                val = ACTIONS_ID["NORTH_WEST"] if delta_y > 0 else ACTIONS_ID["SOUTH_WEST"]
        return val


class Q_BOT(AI_Proj3.BOT_CONFIG):
    def __init__(self, ship, epsilon):
        super(Q_BOT, self).__init__(ship)
        self.old_bot_pos = ()
        self.old_crew_pos = ()
        self.state_1 = self.get_curr_state()
        self.state_2 = np.array([])
        self.is_train = True
        self.epsilon = epsilon

    def get_curr_state(self):
        return np.array([self.ship.get_state((i,j)) for j in range(self.ship.size) for i in range(self.ship.size)]) + np.random.rand(1, self.ship.size**2)/10.0

    def make_action(self, action_no):
        move = self.local_all_moves[action_no]
        next_pos = (self.local_bot_pos[0] + move[0], self.local_bot_pos[1] + move[1])
        self.action_result = AI_Proj3.CLOSED_CELL
        if 0 < next_pos[0] < self.ship.size and 0 < next_pos[1] < self.ship.size:
            state = self.ship.get_state(next_pos)
            if state != AI_Proj3.CLOSED_CELL and state != AI_Proj3.CREW_CELL:
                self.make_bot_move(next_pos)
            self.action_result = state

    def get_loss(self):
        bot_pos = self.local_bot_pos
        crew_pos = self.local_crew_pos
        self.state_2 = self.get_curr_state()
        self.tensor_2 = torch.from_numpy(self.state_2).float()
        reward = self.ship.time_lookup[bot_pos[0]][bot_pos[1]][crew_pos[0]][crew_pos[1]]
        # if self.action_result & (AI_Proj3.CLOSED_CELL | AI_Proj3.CREW_CELL):
        #     reward *= 2
        # elif self.local_crew_pos == self.ship.teleport_cell:
        #     reward = 0
        # else:
        #     reward = 1

        if self.local_crew_pos != self.ship.teleport_cell:
            with torch.no_grad():
                possibleQs = self.ship.q_model(self.tensor_2)
                best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_crew_pos][AI_Proj3.BEST_MOVE]
                best_action = self.ship.get_action(self.local_bot_pos, self.best_move)

            # maxQ = torch.max(possibleQs)
            maxQ = possibleQs.squeeze()[best_action].item()
            newQ = GAAMA*maxQ - reward
        else:
            newQ = 0
        self.policy_reward = torch.Tensor([newQ]).detach()
        print(self.ship.q_model(self.tensor_1), self.ship.q_model(self.tensor_2), self.policy_reward, self.policy_action)
        return self.ship.loss_fn(self.policy_action, self.policy_reward)

    def process_q_learn(self):
        self.tensor_1 = torch.from_numpy(self.state_1).float()
        self.q_vals = self.ship.q_model(self.tensor_1)
        self.best_move = self.ship.best_policy_lookup[self.local_bot_pos][self.local_crew_pos][AI_Proj3.BEST_MOVE]
        self.best_action = self.ship.get_action(self.local_bot_pos, self.best_move)
        if random.uniform(0, 1) < self.epsilon:
            action_no = np.random.randint(0,9)
        else:
            action_no = np.argmax(self.q_vals.data.numpy())

        self.make_action(action_no)
        # loss = self.ship.loss_fn(q_vals.squeeze()[action_no], q_vals.squeeze()[best_action])
        is_rescued = self.move_crew()
        self.policy_action = self.q_vals.squeeze()[action_no]
        loss = self.get_loss()
        self.ship.optimizer.zero_grad()
        loss.backward()
        self.ship.losses.append(loss.item())
        self.ship.optimizer.step()
        # self.state_1 = self.get_curr_state()
        self.state_1 = self.state_2
        return is_rescued

    def move_bot(self):
        tensor_1 = torch.from_numpy(self.state_1).float()
        q_vals = self.ship.q_model(tensor_1)
        action_no = np.argmax(q_vals.data.numpy())
        self.policy_action = q_vals.squeeze()[action_no]
        self.make_action(action_no)

    def move_crew(self):
        ret_val = super(Q_BOT, self).move_crew()
        if self.is_train:
            return ret_val

        # loss = self.get_loss()
        # loss.backward()
        # self.ship.losses.append(loss.item())
        # self.state_1 = self.state_2
        self.state_1 = self.get_curr_state()
        return ret_val

    def train_rescue(self):
        self.is_train = True
        self.total_moves = 0
        if self.ship.get_state(self.local_crew_pos) & AI_Proj3.TELEPORT_CELL:
            return self.total_moves, AI_Proj3.SUCCESS

        while(True):
            self.total_moves += 1
            self.visualize_grid(False)
            if self.process_q_learn():
                self.visualize_grid()
                return self.total_moves, AI_Proj3.SUCCESS

            if self.total_moves > 10000:
                self.visualize_grid()
                return self.total_moves, AI_Proj3.FAILURE

    def test_rescue(self):
        self.is_train = False
        return self.start_rescue()

def t_bot(ship, is_train = True):
    epochs = MAX_TRAIN if is_train else MAX_TEST
    avg_moves = AI_Proj3.DETAILS()
    epsilon = 1.0
    for iters in range(epochs):
        # print(iters, end='\r')
        q_bot = Q_BOT(ship, epsilon)
        if is_train:
            moves, result = q_bot.train_rescue()
        else:
            moves, result = q_bot.test_rescue()

        if result:
            avg_moves.update_min_max(moves)
            avg_moves.s_moves += moves
            avg_moves.success += 1
        else:
            avg_moves.f_moves += moves
            avg_moves.failure += 1

        avg_moves.distance += AI_Proj3.get_manhattan_distance(ship.bot_pos, ship.crew_pos)
        avg_moves.dest_dist += AI_Proj3.get_manhattan_distance(ship.crew_pos, ship.teleport_cell)
        del q_bot

        if epsilon > 0.1:
            epsilon -= (1/epochs)
        ship.reset_grid()

    avg_moves.get_avg(epochs)
    print(("%18s %18s %18s %18s %18s %18s %18s %18s" % (avg_moves.s_moves, avg_moves.success, avg_moves.min_success, avg_moves.max_success, avg_moves.f_moves, avg_moves.failure, avg_moves.distance, avg_moves.dest_dist)))

def single_sim(ship):
    final_data = AI_Proj3.run_sim([range(0, MAX_TRAIN), ship])

    AI_Proj3.print_header(MAX_TRAIN)
    for itr in range(AI_Proj3.TOTAL_CONFIGS):
        AI_Proj3.print_data(final_data, itr, MAX_TRAIN)

def single_run():
    ship = LEARN_CONFIG()
    ship.perform_initial_calcs()
    t_bot(ship)
    ship.print_losses()
    t_bot(ship, False)
    ship.print_losses()
    single_sim(ship)

if __name__ == '__main__':
    begin = time()
    single_run()
    end = time()
    print(end-begin)
