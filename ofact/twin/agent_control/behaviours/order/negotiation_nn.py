from ofact.twin.change_handler.Observer import Observer
from ofact.helpers import Singleton
from projects.bicycle_world.settings import PROJECT_PATH
from projects.bicycle_world.scenarios.current.Train_Settings import get_weights, get_n_sa_list
from torch import nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd



class ExperienceReplay(Dataset,metaclass=Singleton):
    def __init__(self, model, max_memory=200, gamma=0.95, transform=None, target_transform=None):
        self.model = model
        self.memory = []
        self.max_memory = max_memory
        self.gamma = gamma
        self.transform = transform
        self.target_transform = target_transform

    def remember(self, experience, game_over):
        # Save a state to memory
        self.memory.append([experience, game_over])
        # We don't want to store infinite memories, so if we have too many, we just delete the oldest one
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def update_model(self, model):
        self.model = model

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        s, a, r, s_new = self.memory[idx][0]
        goal_state = self.memory[idx][1]
        features = np.array(s)
        # init labels with old prediction (and later overwrite action a)
        label = self.model[s][0]
        if goal_state:
            label[a] = r
        else:
            label[a] = r + self.gamma * max(self.model[s_new][0])

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        features = torch.from_numpy(features).float().to(device)
        label = torch.from_numpy(label).float().to(device)

        return features, label

class DeepQTable(nn.Module):

    def __init__(self, number_of_states, number_of_actions, Optimizer, loss_fn, transform):
        super(DeepQTable, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(number_of_states, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, number_of_actions)
        )
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.to(self.device)
        self.optimizer = Optimizer(self.parameters())
        self.loss_fn = loss_fn
        self.transform=transform


    def __getitem__(self, state):
        state = torch.tensor(self.transform(np.array(state))).float().to(self.device)
        self.model.eval()
        #print(self.device)
        prediction = self(state[None,]).cpu().detach().numpy()
        self.model.train()
        return prediction

    def __setitem__(self, state, value):
        # ignoring setting to values
        pass

    def forward(self, x):
        logics= self.model(x)
        return logics


    def perform_training(self, dataloader):
        loss_history = []
        (X, y) = next(iter(dataloader))
        # Compute prediction and loss
        pred = self(X)
        loss = self.loss_fn(pred, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_history.append(loss.item())
        return loss_history



    def save_model(self, file):
        torch.save(self.state_dict(), file)

    def load_model(self, file):
        self.load_state_dict(torch.load(file))


class DeepQLearningAgent():

    def __init__(self, q_table=None, N_sa=None, gamma=0.95, max_N_exploration=1, R_Max=100, batch_size=25,
                 Optimizer=torch.optim.Adam, loss_fn=nn.MSELoss(), ModelClass=DeepQTable):
        self.num_actions = 6
        self.actions=self.set_actions()
        self.num_states = 4+self.num_actions
        min_values = 0
        max_values = 100
        self.transform = lambda x: (x - min_values) / (max_values - min_values)
        self.q_table = self.create_model(Optimizer, loss_fn, self.transform, ModelClass)
        weights=get_weights()
        self.N_sa = get_n_sa_list()[0]
        if len(weights)>0:
            self.q_table.load_model('model_weights.pth')
            self.q_table.optimizer.load_state_dict(torch.load('optimizer.pth'))
            self.N_sa = torch.load('na_dict.pt',weights_only=False)
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max
        self.batch_size = batch_size
        self.experience_replay = ExperienceReplay(self.q_table,transform=self.transform)
        self.loss_history = []
        self.observer=Observer()
        self.possible_time = None
        self.s_new = None
        self.last_output=None
        self.last_pe_id=None
        self.last_vap={}
        self.reward_penelty=1/self.num_actions
        self.failed_negotiation=0
        self.amount_negotiated=0
        self.answer=self.get_answer_dict()
        self.observer.set_observation(self.actions[self.num_actions-1]['end time'])
        self.max_time=426
        self.num_order_normalization_factor = 3151

    def track_no_training(self, value_added_process, last_pe):
        if last_pe.identification == self.last_pe_id:
            self.failed_negotiation += 1
        self.last_pe_id = last_pe.identification
        self.amount_negotiated += 1

    def get_answer_dict(self):
        answer = {}
        for i in range(self.num_actions):
            answer[i] = {'True': 0, 'False': 0}
        return answer
    def create_model(self, Optimizer, loss_fn, transform, ModelClass):
        return ModelClass(self.num_states, self.num_actions, Optimizer, loss_fn, transform)


    def set_actions(self):
        actions = {}

        start_time=np.timedelta64(451,'s') #längster Process ca. 7:30 min
        for i in range(self.num_actions):
            actions[i] = {}
            start_time= start_time - np.timedelta64(450, 's')
            actions[i]['start time']= start_time
            actions[i]['end time']=start_time+ np.timedelta64(900, 's')
            start_time=actions[i]['end time']
        """
        for i in range(self.num_actions):
            actions[i] = {}
            actions[i]['start time']= np.timedelta64(1,'s')
            actions[i]['end time']=np.timedelta64((i+1)*60*10,'s')
        """
        """
        #implementation in case the agent should execute another process for i in range(self.num_actions -1)
        actions[i+1]['start time']=None
        actions[i+1]['end time']=None
        """
        return actions

    """
    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)
        if len(self.experience_replay) > self.batch_size:
            train_loader = DataLoader(self.experience_replay, batch_size=self.batch_size, shuffle=True)
            self.loss_history += self.q_table.perform_training(train_loader)
            self.loss_history = self.loss_history[-100:]
    """

    def update_q_values(self, s, a, r, s_new, is_goal_state):
        self.experience_replay.remember((s, a, r, s_new), is_goal_state)

        if len(self.experience_replay) > self.batch_size:
            train_loader = DataLoader(
                self.experience_replay,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0  # Kein multiprocessing → stabiler Speicherverbrauch
            )
            new_losses = self.q_table.perform_training(train_loader)
            self.loss_history += new_losses[-10:]  # Nur letzte 10 behalten
            self.loss_history = self.loss_history[-100:]

    def train(self,value_added_process,last_pe):
        resource_used = value_added_process.resource_controller.resource_model.resource_groups[0].resources
        if last_pe.identification==self.last_pe_id:
            negotiation_successful=False
        else:
            negotiation_successful=True
        self.last_pe_id=last_pe.identification
        r = self.get_eval(negotiation_successful)
        s = self.s_new
        self.s_new = tuple(self.get_current_state(resource_used,value_added_process))
        if self.s_new not in self.N_sa.keys():
            self.N_sa[self.s_new] = np.zeros(len(self.actions))
            self.q_table[self.s_new] = np.zeros(len(self.actions))
        if self.possible_time is not None:
            self.N_sa[s][self.last_output] += 1
            self.update_q_values(s, self.last_output, r, self.s_new, self.is_goal_state())
        if self.is_goal_state():
            return self.q_table, self.N_sa
        self.possible_time = self.choose_GLIE_action(self.q_table[self.s_new], self.N_sa[self.s_new])
        print(f'{resource_used[0].name}: {self.s_new} = {self.last_output}')
        if value_added_process.identification in self.last_vap:
            history = self.last_vap[value_added_process.identification]
            history.append(self.last_output)
            if len(history) > 10:
                history.pop(0)  # Ältesten Eintrag entfernen
        else:
            self.last_vap[value_added_process.identification] = [self.last_output]
        return self.possible_time

    def get_eval(self, answer):
        """
        old preference: preference calculate through meta heuristic (Adrian)
        nn_preference: preference calculated from the neuronal network

        reward: Reward has two factors that influence the result. Firstly, whether the result has led to a positive
        answer and secondly, whether there is an appointment that would be more suitable. Whether there is a previous value is inferred from the meta heuristic

        return reward
        """
        self.reward=0
        if self.last_output is not None:
            if answer == True:
                self.reward+=1
                self.reward+=(1-self.reward_penelty*self.last_output)*4
                self.answer[self.last_output]['True']+=1
            else:
                self.reward-=1
                self.reward-=(1-self.reward_penelty*self.last_output)
                s_new = self.s_new[2:]
                if s_new[self.last_output] == 1:
                    self.reward-=3
                self.failed_negotiation+=1
                self.answer[self.last_output]['False'] += 1
        else:
            if answer == True:
                self.reward += 1
            else:
                self.reward -= 1
                self.failed_negotiation+=1

        """
        if old_preference == nn_preference:
            self.reward+=1
        else:
            self.reward-=1
        """
        return self.reward


    def get_current_state(self,resource_used,value_added_process):
        self.amount_negotiated+=1
        if resource_used[0].name == 'Warehouse':
            print('Debug')
        utilisation_df=self.observer.get_utilisation()
        workstation, num_order=self.observer.get_workstation(resource_used)
        if not utilisation_df.empty:
            agv_df = utilisation_df[utilisation_df.index.str.startswith("Main Part")]
            mean_utili = agv_df.mean()
            utilisation_df['Agv Mean']=mean_utili
            if workstation.name in utilisation_df.index:
               current_state=utilisation_df[['Agv Mean',workstation.name]].tolist() # 'Main Warehouse'
               #current_state = [int(utilisation_df[workstation.name])]
            else:
                current_state=utilisation_df[['Agv Mean']].tolist() # 'Main Warehouse'
                current_state=current_state+[0]
                #current_state=list(np.zeros(1))
        else:
            #current_state=list(np.zeros(3))
            current_state = list(np.zeros(2))
        current_state=[round(x) / 100
                       for x in current_state]

        current_state.append(num_order / self.num_order_normalization_factor)
        try:
            time=value_added_process.lead_time_controller.process_time_model.mue
            time=time/self.max_time
            current_state.append(time)
        except:
            time = value_added_process.lead_time_controller.process_time_model.value
            time = time / self.max_time
            current_state.append(time)
        if value_added_process.identification in self.last_vap:
            last_output=list(np.zeros(self.num_actions))
            for i in self.last_vap[value_added_process.identification]:
                last_output[i]=1
            current_state=current_state+ last_output
        else:
            current_state = current_state + list(np.zeros(self.num_actions))
        return current_state




    def is_goal_state(self):
        pass

    def choose_GLIE_action(self, q_values, N_s):
        exploration_values = np.ones_like(q_values) * self.R_Max
        # which state/action pairs have been visited sufficiently
        no_sufficient_exploration = N_s < self.max_N_exploration
        # turn cost to a positive number
        q_values_pos = self.R_Max / 2 + q_values
        # select the relevant values (q or max value)
        max_values = np.maximum(exploration_values * no_sufficient_exploration, q_values_pos)
        # assure that we do not dived by zero
        if max_values.sum() == 0:
            probabilities = np.ones_like(max_values) / max_values.size
        else:
            probabilities = max_values / max_values.sum()
        # select action according to the (q) values

        if np.random.random() < (self.max_N_exploration+0.00001)/(np.max(N_s)+0.00001):
            action = np.random.choice(range(self.num_actions), p=probabilities.flatten())
            self.last_output = action
            action=self.actions[action]
        else:
            action_indexes = np.argwhere(probabilities == np.amax(probabilities))
            #action_indexes.shape = (action_indexes.shape[0])
            action_index = np.random.choice(action_indexes[:,1])
            self.last_output = action_index
            action = self.actions[action_index]
        return action

    def get_time_periods(self,value_added_process,last_pe):
        resource_used = value_added_process.resource_controller.resource_model.resource_groups[0].resources
        if last_pe.identification==self.last_pe_id:
            self.failed_negotiation+=1
        self.last_pe_id = last_pe.identification
        self.s_new = tuple(self.get_current_state(resource_used,value_added_process))
        a=self.output_cal()
        return self.actions[a]

    def output_cal(self):
        a=self.q_table[self.s_new]
        action_indexes = np.argwhere(a == np.amax(a))
        self.last_output= int(action_indexes[:,1])
        return self.last_output