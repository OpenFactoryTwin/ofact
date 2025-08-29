from ofact.twin.change_handler.Observer import Observer
from projects.bicycle_world.scenarios.speed.Train_Settings import get_weights, get_n_sa_list
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.distributions import Categorical
import torch
import numpy as np


class ExperienceReplay(Dataset):
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
        #goal_state = self.memory[idx][1]
        #features = np.array(s)
        # init labels with old prediction (and later overwrite action a)
        """
        label = self.model[s][0]
        if goal_state:
            label[a] = r
        else:
            label[a] = r + self.gamma * max(self.model[s_new][0])
        """
        logits = self.model[s][0]
        features = np.array(s)
        label = (a, r)  # a: int, r: float
        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            label = self.target_transform(label)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        features = torch.from_numpy(features).float().to(device)
        action, reward = label
        return features, torch.tensor(action), torch.tensor(reward, dtype=torch.float32)

        return features, label

class DeepQTable(nn.Module):

    def __init__(self, number_of_states, number_of_actions, Optimizer, loss_fn, transform):
        super(DeepQTable, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(number_of_states, 64),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(32, number_of_actions),
            nn.Softmax(dim=-1)
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

    """
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
    """

    def perform_training(self, dataloader):
        loss_history = []
        for states, actions, rewards in dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)

            probs = self(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)

            loss = -torch.mean(log_probs * rewards)

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

    def __init__(self, model=None, N_sa=None, gamma=0.95, max_N_exploration=2, R_Max=100, batch_size=25,
                 Optimizer=torch.optim.Adam, loss_fn=None, ModelClass=DeepQTable):
        self.num_actions = 12
        self.actions=self.set_actions()
        self.num_states = 4+self.num_actions
        min_values = 0
        max_values = 100
        self.transform = lambda x: (x - min_values) / (max_values - min_values)
        self.model = self.create_model(Optimizer,loss_fn ,self.transform, ModelClass)
        weights=get_weights()
        self.N_sa = get_n_sa_list()[0]
        if len(weights)>0:
            self.model.load_model('model_weights.pth')
            self.model.optimizer.load_state_dict(torch.load('optimizer.pth'))
            self.N_sa = torch.load('na_dict.pt',weights_only=False)
        self.gamma = gamma
        self.max_N_exploration = max_N_exploration
        self.R_Max = R_Max
        self.batch_size = batch_size
        self.experience_replay = ExperienceReplay(self.model,transform=self.transform)
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
        self.num_processe_normalization_factor = 3151 #kann der kleinste process ausgeführt werden im Observer zeitraum
        self.round_count=0
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
            new_losses = self.model.perform_training(train_loader)
            self.loss_history += new_losses[-10:]  # Nur letzte 10 behalten
            self.loss_history = self.loss_history[-100:]

    def train(self,value_added_process,last_pe):
        resource_used = value_added_process.resource_controller.resource_model.resource_groups[0].resources
        if value_added_process.identification in self.last_vap.keys():
            negotiation_successful=False
            self.round_count += 1
        else:
            negotiation_successful=True
            self.round_count = 0
        self.last_pe_id=value_added_process.identification
        r = self.get_eval_v2(negotiation_successful)
        s = self.s_new
        self.s_new = tuple(self.get_current_state(resource_used,value_added_process))
        if self.s_new not in self.N_sa.keys():
            self.N_sa[self.s_new] = np.zeros(len(self.actions))
            self.model[self.s_new] = np.zeros(len(self.actions))
        if self.possible_time is not None:
            self.N_sa[s][self.last_output] += 1
            self.update_q_values(s, self.last_output, r, self.s_new, self.is_goal_state())
        if self.is_goal_state():
            return self.model, self.N_sa
        #self.possible_time = self.choose_GLIE_action(self.q_table[self.s_new], self.N_sa[self.s_new])
        self.possible_time = self.choose_action_from_policy(self.s_new)
        print(f'{resource_used[0].name}: {self.s_new} = {self.last_output},  round: {self.round_count}, vap: {value_added_process.identification}')
        if value_added_process.identification in self.last_vap:
            self.last_vap[value_added_process.identification][self.last_output]=1
        else:
            self.last_vap[value_added_process.identification]=np.zeros(self.num_actions)
            self.last_vap[value_added_process.identification][self.last_output] = 1
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

    def get_eval_v2(self, negotiation_successful):
        # 1. Success Reward
        if self.last_output is None:
            return 0
        reward_success = 1.0 if negotiation_successful else -1.0

        # 2. Penalty für Wiederholung (wenn selbe Zeit wie beim letzten Mal gewählt wurde)
        s_new = self.s_new[2:]
        repeated_choice = s_new[self.last_output] == 1
        reward_penalty = -1.0 if repeated_choice else 0.0

        # 3. Effizienz: Früher gewählte Slots sind besser
        # Annahme: self.last_output ist Index des gewählten Slots, lower index = früher
        num_slots = len(s_new)
        normalized_time = self.last_output * 5 / (num_slots - 1)
        reward_efficiency = 5.0 - normalized_time  # früher = 1.0, später = 0.0

        # 4. Diversity: Selten gewählte Aktionen werden belohnt
        usage_count = self.N_sa[self.s_new][self.last_output]
        reward_diversity = 1.0 / (1.0 + usage_count)  # z. B. 1.0 bei erster Wahl, abnehmend

        # 5. Gewichtete Kombination
        reward = (
                1.0 * reward_success +
                0.5 * reward_efficiency +
                0.3 * reward_diversity +
                0.7 * reward_penalty
        )

        return reward



    def get_current_state(self,resource_used,value_added_process):
        self.amount_negotiated+=1
        resource_name=resource_used[0].name
        if resource_used[0].name == 'Warehouse':
            resource_name='Main Warehouse'
        utilisation_df=self.observer.get_utilisation()
        num_processe=self.observer.get_num_process(resource_name)
        #workstation= self.observer.get_workstation(resource_used)
        if not utilisation_df.empty:
            agv_df = utilisation_df[utilisation_df.index.str.startswith("Individual Part")]
            mean_utili = agv_df.values.mean()
            if resource_name in utilisation_df.index:
                current_state=[round(mean_utili,2),round(utilisation_df.at[resource_name],2)]# 'Main Warehouse'
                #current_state=[round(utilisation_df.at[resource_name],2)]
            else:
                current_state=[round(mean_utili,2), 0]
                #current_state=[0]
        else:
            #current_state=[0]
            current_state = list(np.zeros(2))
        current_state=current_state+[num_processe / self.num_processe_normalization_factor]
        time_model=value_added_process.lead_time_controller.process_time_model
        if hasattr(time_model, 'mue'):
            time = time_model.mue / self.max_time
        else:
            time = time_model.value / self.max_time

        current_state = current_state + [time]
        if value_added_process.identification in self.last_vap:
            last_output = self.last_vap[value_added_process.identification]
        else:
            last_output = [0] * self.num_actions

        current_state += list(last_output)
        return current_state




    def is_goal_state(self):
        pass

    def choose_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(self.transform(np.array(state))).float().to(self.model.device)
            probs = self.model(state_tensor.unsqueeze(0))
            if self.round_count >0:
                print('action manipulation')
                old_probs=probs.clone()
                probs[torch.tensor(self.last_vap[self.last_pe_id]).unsqueeze(0) == 1] = 0
                if probs.sum().item()==0:
                    probs=old_probs
            T = max(0.1, 1.0 / (1.0 + self.round_count))  # sinkende Temperatur
            probs = torch.pow(probs, 1 / T)
            probs /= probs.sum()
            dist = torch.distributions.Categorical(probs)
            action_index = dist.sample().item()
            self.last_output = action_index
            action = self.actions[action_index]
        return action

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

#        if np.any(N_s < self.max_N_exploration):
            # => Es gibt mindestens eine Aktion, die noch nicht ausreichend exploriert wurde
#            action_index = np.random.choice(range(self.num_actions), p=probabilities.flatten())
#            self.last_output = action_index
#            action=self.actions[action_index]

        if np.random.random() < (self.max_N_exploration+0.00001)/(np.max(N_s)+0.00001):
            action_index = np.random.choice(range(self.num_actions), p=probabilities.flatten())
            self.last_output = action_index
            action=self.actions[action_index]

        else:
            action_indexes = np.argwhere(probabilities == np.amax(probabilities))
            #action_indexes.shape = (action_indexes.shape[0])
            action_index = np.random.choice(action_indexes[:,1])
            self.last_output = action_index
            action = self.actions[action_index]
        return action


    #deployment
    def get_time_periods(self,value_added_process,last_pe):
        resource_used = value_added_process.resource_controller.resource_model.resource_groups[0].resources
        if value_added_process.identification in self.last_vap.keys():
            self.failed_negotiation+=1
            negotiation_successful = False
        else:
            negotiation_successful = True
        self.last_pe_id = value_added_process.identification
        self.s_new = tuple(self.get_current_state(resource_used,value_added_process))
        a=self.output_cal(negotiation_successful)
        if value_added_process.identification in self.last_vap:
            self.last_vap[value_added_process.identification][a]=1
        else:
            self.last_vap[value_added_process.identification]=np.zeros(self.num_actions)
            self.last_vap[value_added_process.identification][a] = 1
        return self.actions[a]

    def output_cal(self,negotiation_successful):
        a=self.model[self.s_new]
        if not negotiation_successful:
            a[torch.tensor(self.last_vap[self.last_pe_id]).unsqueeze(0) == 1] = 0
        action_indexes = np.argwhere(a == np.amax(a))
        self.last_output= int(min(action_indexes[:,1]))
        return self.last_output


#------------------------------------------------------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.next_states = []

    def add(self, state, action, reward, log_prob, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.next_states.append(next_state)

    def size(self):
        return len(self.states)



class DeepActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, Optimizer):
        super(DeepActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer_actor = Optimizer(self.actor.parameters(), lr=1e-3)
        self.optimizer_critic = Optimizer(self.critic.parameters(), lr=1e-3)

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def get_optimizer(self):
        optimizer={}
        optimizer['optimizer_actor'] = self.optimizer_actor.state_dict()
        optimizer['optimizer_critic'] = self.optimizer_critic.state_dict()
        return optimizer



class PolicyLearningAgent():
    #actor critic
    def __init__(self, model=None, N_sa=None, gamma=0.95, max_N_exploration=2, R_Max=100, batch_size=25,
                 Optimizer=torch.optim.Adam, loss_fn=None):
        self.num_actions = 12
        self.actions = self.set_actions()
        self.num_state_without_actions =4
        self.num_states = self.num_state_without_actions + self.num_actions

        self.model = DeepActorCritic(self.num_states, self.num_actions, Optimizer)
        weights = get_weights()
        self.N_sa = get_n_sa_list()[0]
        if len(weights) > 0:
            self.model.load_model('model_weights.pth')
            checkpoint = torch.load('optimizer.pth')
            self.model.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
            self.model.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        self.buffer = RolloutBuffer(capacity=32)
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_history = []
        self.observer = Observer()
        self.possible_time = None
        self.s_new = None
        self.last_output = None
        self.last_pe_id = None
        self.last_vap = {}
        self.reward_penelty = 1 / self.num_actions
        self.failed_negotiation = 0
        self.amount_negotiated = 0
        self.answer = self.get_answer_dict()
        self.observer.set_observation(self.actions[self.num_actions - 1]['end time'])
        self.max_time = 426
        self.num_processe_normalization_factor = 3151  # kann der kleinste process ausgeführt werden im Observer zeitraum
        self.round_count = 0


    def get_answer_dict(self):
        answer = {}
        for i in range(self.num_actions):
            answer[i] = {'True': 0, 'False': 0}
        return answer

    def set_actions(self):
        actions = {}

        start_time=np.timedelta64(451,'s') #längster Process ca. 7:30 min
        for i in range(self.num_actions):
            actions[i] = {}
            start_time= start_time - np.timedelta64(450, 's')
            actions[i]['start time']= start_time
            actions[i]['end time']=start_time+ np.timedelta64(900, 's')
            start_time=actions[i]['end time']

        return actions

    def update_network_from_buffer(self, clip_eps=0.2, entropy_coeff=0.01, epochs=4):
        self.model.train()

        states = torch.stack(self.buffer.states)
        actions = torch.tensor(self.buffer.actions)
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32)
        log_probs_old = torch.stack(self.buffer.log_probs)
        next_states = torch.stack(self.buffer.next_states)

        # Compute values
        _, values = self.model(states)
        _, next_values = self.model(next_states)
        returns = rewards + self.gamma * next_values.squeeze()
        advantages = returns - values.squeeze()

        # Vorteil normalisieren
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = advantages.detach()
        returns = returns.detach()
        log_probs_old = log_probs_old.detach()
        for _ in range(epochs):  # Mehrfach über Buffer iterieren!
            probs, values = self.model(states)
            dist = Categorical(probs)
            log_probs = dist.log_prob(actions)

            # Clipped Surrogate Objective
            ratio = torch.exp(log_probs - log_probs_old.detach())
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy-Bonus für Exploration
            entropy = dist.entropy().mean()
            actor_loss -= entropy_coeff * entropy

            # Critic Loss
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()

            loss = actor_loss + critic_loss

            self.model.optimizer_actor.zero_grad()
            self.model.optimizer_critic.zero_grad()
            loss.backward()
            self.model.optimizer_actor.step()
            self.model.optimizer_critic.step()

        self.loss_history.append(loss.item())

    def train(self,value_added_process,last_pe):
        resource_used = value_added_process.resource_controller.resource_model.resource_groups[0].resources
        if value_added_process.identification in self.last_vap.keys():
            negotiation_successful=False
            self.round_count += 1
        else:
            negotiation_successful=True
            self.round_count = 0
        self.last_pe_id=value_added_process.identification
        self.s=self.s_new
        self.reward=self.get_eval_v2(negotiation_successful)
        self.s_new = torch.tensor(self.get_current_state(resource_used,value_added_process),dtype=torch.float32)
        if self.s is not None:
            self.buffer.add(self.s,self.last_output, self.reward,self.last_log_prob, self.s_new)
        action, log = self.select_action(self.s_new)
        self.last_output = action
        self.last_log_prob=log
        self.possible_time = self.actions[action]
        if self.buffer.size() >=self.buffer.capacity:
            self.update_network_from_buffer()
            self.buffer.reset()
        print(f'{resource_used[0].name}: {self.s_new} = {self.last_output},  round: {self.round_count}, vap: {value_added_process.identification} agent: {last_pe}')
        if value_added_process.identification in self.last_vap:
            self.last_vap[value_added_process.identification][self.last_output]=1
        else:
            self.last_vap[value_added_process.identification]=np.zeros(self.num_actions)
            self.last_vap[value_added_process.identification][self.last_output] = 1
        return self.possible_time

    def get_eval_v2(self, negotiation_successful):
        if self.last_output is None:
            return 0

        # 1. Erfolg
        reward_success = 1.0 if negotiation_successful else -1.0

        # 2. Wiederholung vermeiden
        s_new = self.s_new[self.num_state_without_actions:]
        efficiency_factor = 1.0 - torch.mean(self.s_new[:2])  # globale Auslastung

        repeated_choice = s_new[self.last_output] == 1
        reward_penalty = -1.0 if repeated_choice else 0.0

        # 3. Effizienz-Bias
        normalized_time = self.last_output * 5 / (self.num_actions - 1)
        reward_efficiency = (5.0 - normalized_time) * efficiency_factor

        # 4. Positions-Strafe (wenn Vorgänger nicht gewählt)
        if self.last_output > 0:
            prev_was_requested = s_new[self.last_output - 1] == 1
        else:
            prev_was_requested = True

        if not prev_was_requested:
            position_penalty = normalized_time * 5.0 * efficiency_factor
        else:
            position_penalty = 0.0

        # 5. Gesamt
        reward = (
                1.0 * reward_success +
                0.9 * reward_efficiency +
                0.7 * reward_penalty -  # penalty = -1 wenn Wiederholung
                position_penalty
        )

        return reward



    def get_current_state(self,resource_used,value_added_process):
        self.amount_negotiated+=1
        resource_name=resource_used[0].name
        if resource_used[0].name == 'Warehouse':
            resource_name='Main Warehouse'
        self.observer.update_kpi()
        utilisation_df=self.observer.get_utilisation()
        num_processe=self.observer.get_num_process(resource_name)
        #workstation= self.observer.get_workstation(resource_used)
        if not utilisation_df.empty:
            agv_df = utilisation_df[utilisation_df.index.str.startswith("Individual Part")]
            mean_utili = agv_df.values.mean()
            if resource_name in utilisation_df.index:
                current_state=[round(mean_utili,2),round(utilisation_df.at[resource_name],2)]# 'Main Warehouse'
                #current_state=[round(utilisation_df.at[resource_name],2)]
            else:
                current_state=[round(mean_utili,2), 0]
                #current_state=[0]
        else:
            #current_state=[0]
            current_state = list(np.zeros(2))
        current_state=current_state+[num_processe / self.num_processe_normalization_factor]
        time_model=value_added_process.lead_time_controller.process_time_model
        if hasattr(time_model, 'mue'):
            time = time_model.mue / self.max_time
        else:
            time = time_model.value / self.max_time

        current_state = current_state + [time]
        if value_added_process.identification in self.last_vap:
            last_output = self.last_vap[value_added_process.identification]
        else:
            last_output = [0] * self.num_actions

        current_state += list(last_output)
        current_state = [float(x) for x in current_state]
        return current_state

    def is_goal_state(self):
        pass

    def select_action(self, state):
        self.model.eval()
        with torch.no_grad():
            probs, _ = self.model(state)
            if self.round_count >0:
                print('action manipulation')
                old_probs=probs.clone()
                probs[torch.tensor(self.last_vap[self.last_pe_id]) == 1] = 0
                if probs.sum().item()==0:
                    probs=old_probs
            T = max(0.1, 1.0 / (1.0 + self.round_count))  # sinkende Temperatur
            probs = torch.pow(probs, 1 / T)
            probs /= probs.sum()
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        self.model.train()
        self.last_action = action.item()
        return action.item(), log_prob.cpu()


    # deployment
    def get_time_periods(self, value_added_process, last_pe):
        resource_used = value_added_process.resource_controller.resource_model.resource_groups[0].resources
        if value_added_process.identification in self.last_vap.keys():
            self.failed_negotiation += 1
            negotiation_successful = False
        else:
            negotiation_successful = True
        self.last_pe_id = value_added_process.identification
        self.s_new = torch.tensor(self.get_current_state(resource_used, value_added_process))
        a = self.output_cal(negotiation_successful)
        if value_added_process.identification in self.last_vap:
            self.last_vap[value_added_process.identification][a] = 1
        else:
            self.last_vap[value_added_process.identification] = np.zeros(self.num_actions)
            self.last_vap[value_added_process.identification][a] = 1
        return self.actions[a]

    def output_cal(self, negotiation_successful):
        a = self.model(self.s_new)
        if not negotiation_successful:
            a[0][torch.tensor(self.last_vap[self.last_pe_id]) == 1] = 0
        action_indexes_tuple = torch.where(a[0] == torch.max(a[0]))
        action_indexes_tensor = action_indexes_tuple[0]
        action_indexes = action_indexes_tensor.item()

        self.last_output = action_indexes
        return self.last_output
