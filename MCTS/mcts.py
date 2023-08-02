from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from multiprocessing import Manager
from threading import Lock, Condition
import concurrent.futures.thread

import numpy as np
# import cchess_alphazero.environment.static_env as senv
# from cchess_alphazero.config import Config
# from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from time import time, sleep
import os
import gc 
import sys
import copy
import ipdb

logger = getLogger(__name__)

class VisitState:
    def __init__(self):
        self.a = defaultdict(ActionState)                            # key: action, value: ActionState
        self.sum_n = 0                                               # visit count
        self.visit = {'hist':[], 'obs':[], 'root':[], 'actions':[]}  # thread id that has visited this state
        self.p = None                                                # policy of this state
        self.legal_moves = None                                      # all leagal moves of this state
        self.waiting = False                                         # is waiting for NN's predict
        self.w = None


class ActionState:
    def __init__(self):
        self.n = 0      # N(s, a) : visit count
        self.w = 0      # W(s, a) : total action value
        self.q = 0      # Q(s, a) = N / W : action value
        self.p = 0      # P(s, a) : prior probability

class MCTS:
    def __init__(self, env, config, search_tree=None, pipes=None, K=None, debugging=False, use_history=False, training=False):
        self.env = env
        self.config = config
        # self.play_config = play_config or self.config.play

        self.labels_n = self.env.action_space[0].n
        self.labels = range(self.labels_n)
        self.move_lookup = {move: i for move, i in zip(self.labels, range(self.labels_n))}
        self.pipe = pipes                   # pipes that used to communicate with CChessModelAPI thread
        self.node_lock = defaultdict(Lock)  # key: state key, value: Lock of that state
        self.use_history = use_history
        self.increase_temp = False
        self.training = training
        self.K = K
        if self.training:
            self.simulation_num_per_move = self.config["simulation_num_per_move_train"]
        else:
            self.simulation_num_per_move = self.config["simulation_num_per_move_test"]
        # print("training state:", self.training, "simulation num:",self.simulation_num_per_move)
        if search_tree is None:
            self.tree = defaultdict(VisitState)  # key: state key, value: VisitState
        else:
            self.tree = search_tree

        self.root_state = None

        # self.enable_resign = enable_resign
        self.debugging = debugging

        self.search_results = {}        # for debug
        self.debug = {}
        # self.side = side

        self.s_lock = Lock()
        self.run_lock = Lock()
        self.q_lock = Lock()            # queue lock
        self.t_lock = Lock()
        self.buffer_planes = []         # prediction queue
        self.buffer_history = []

        self.all_done = Lock()
        self.num_task = 0
        self.done_tasks = 0
        # self.uci = uci
        self.no_act = None       # 不被考虑的动作

        self.job_done = False

        self.executor = ThreadPoolExecutor(max_workers=self.config["search_threads"] + 2)
        self.executor.submit(self.receiver)
        self.executor.submit(self.sender)

    def close(self, wait=True):
        self.job_done = True
        del self.tree
        gc.collect()
        if self.executor is not None:
            self.executor.shutdown(wait=wait)

    def close_and_return_action(self, state, turns, no_act=None):
        self.job_done = True
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            # self.executor = None
            self.executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()
        policy, resign = self.calc_policy(state, turns, no_act)
        if resign:  # resign
            return None
        if no_act is not None:
            for act in no_act:
                policy[self.move_lookup[act]] = 0
        my_action = int(np.random.choice(range(self.labels_n), p=self.apply_temperature(policy, turns)))
        if state in self.debug:
            _, value = self.debug[state]
        else:
            value = 0
        return self.labels[my_action], value, self.done_tasks // 100

    def sender(self):
        '''
        send planes to neural network for prediction
        '''
        limit = 256                 # max prediction queue size
        while not self.job_done:
            self.run_lock.acquire()
            with self.q_lock:
                l = min(limit, len(self.buffer_history))
                if l > 0:
                    t_data = self.buffer_planes[0:l]
                    # logger.debug(f"send queue size = {l}")
                    # print('t_data:',t_data.shape)
                    self.pipe.send(t_data)
                    # print("------5555")
                else:
                    self.run_lock.release()
                    sleep(0.001)

    def receiver(self):
        '''
        receive policy and value from neural network
        '''
        while not self.job_done:
            if self.pipe.poll(0.001):
                rets = self.pipe.recv()
            else:
                continue
            k = 0
            with self.q_lock:
                for ret in rets:
                    # logger.debug(f"NN ret, update tree buffer_history = {self.buffer_history}")
                    # print("5555555555")
                    self.executor.submit(self.update_tree, ret[0], ret[1], self.buffer_history[k])
                    # self.update_tree(ret[0], ret[1], self.buffer_history[k])
                    k = k + 1
                self.buffer_planes = self.buffer_planes[k:]
                self.buffer_history = self.buffer_history[k:]
            self.run_lock.release()

    def action(self, observation, state, num_steps, unallowed_act, depth=None, hist=None, increase_temp=False) -> str:
        self.all_done.acquire(True)
        self.root_state = state  #将每一个搜索开始的节点视为根节点。根节点不一定是初始节点，随着每一次动作的选择，根节点会变化
        self.no_act = list(unallowed_act)
        self.increase_temp = increase_temp
        # if hist and len(hist) >= 5:
        #     hist = hist[-5:]
        done = 0
        # print(state)
        # print(self.tree)
        # exit()
        if state in self.tree:
            # 当下了几次子之后，对于一个新来的state，可能之前的几次下子已经路过该state好几次了
            done = self.tree[state].sum_n
        # if no_act or increase_temp or done == self.play_config.simulation_num_per_move:
        if done == self.simulation_num_per_move:
            # logger.info(f"no_act = {no_act}, increase_temp = {increase_temp}")
            done = 0
        self.done_tasks = done
        self.num_task = self.simulation_num_per_move - done
        if depth:
            self.num_task = depth - done if depth > done else 0
        # if infinite:
        #     self.num_task = 100000
        depth = 0
        start_time = time()
        # MCTS search
        if self.num_task > 0:
            all_tasks = self.num_task
            batch = all_tasks // self.config["search_threads"]
            if all_tasks % self.config["search_threads"] != 0:
                batch += 1
            # logger.debug(f"all_task = {self.num_task}, batch = {batch}")
            # print(self.config["simulation_num_per_move"], all_tasks, done, state)
            for iter in range(batch):
                # print(iter)
                # print(self.no_act)
                self.num_task = min(self.config["search_threads"], all_tasks - self.config["search_threads"] * iter)
                self.done_tasks += self.num_task
                # logger.debug(f"iter = {iter}, num_task = {self.num_task}")
                for i in range(self.num_task):
                    try:
                        self.executor.submit(self.MCTS_search, observation, state, [state], True, self.no_act) #self.no_act = real_hist，之前选择的动作历史之后不允许再次选择
                    except Exception as e:
                        print(e)
                self.all_done.acquire(True)
                # if self.uci and depth != self.done_tasks // 100:
                    # info depth xx pv xxx
                    # depth = self.done_tasks // 100
                    # _, value = self.debug[state]
                    # self.print_depth_info(state, turns, start_time, value, no_act)
        self.all_done.release()

        # policy, resign = self.calc_policy(state, turns, no_act)
        policy = self.calc_policy(state, self.no_act)

        # if resign:  # resign
        #     return None, list(policy)
        if self.no_act is not None:
            for act in self.no_act:
                policy[act-1] = 0

        my_action = int(np.random.choice(range(1,self.labels_n+1), p=self.apply_temperature(policy, num_steps)))
        return my_action, list(policy)#np.array(policy).reshape(1,-1)

    def MCTS_search(self, observation, state, history=[], is_root_node=False, real_hist=None) -> float:
        """
        Monte Carlo Tree Search
        """
        def get_legal_mov(state, actions, K, is_root_node):
            if is_root_node and len(actions)==0: #树的开始，合法动作为所有没被选择的stock id
                legal_moves = list(state)
            elif len(actions) != 0:
                if actions[-1] < self.labels_n/2:
                    legal_moves = [a for a in state if a>actions[-1]]
                else:
                    legal_moves = [a for a in state if a<actions[-1]]

            # if len(actions)>0: 
            #     #获取比上一个动作大的action id并且 legal action id下的子树要满足一定深度
            #     legal_moves = [a for a in state if a>actions[-1] and a<=self.labels_n-K+len(actions)] 
            # else: #如果之前没做过选择
            #     legal_moves = [a for a in state if a<=self.labels_n-K+len(actions)]

            return legal_moves

        actions = copy.deepcopy(real_hist)  #real_hist 表示在当前搜索节点之前执行的动作
        obs = copy.deepcopy(observation)
        while True:
            # logger.debug(f"start MCTS, state = {state}, history = {history}")
            # print("execute search")
            # print(actions)
            # 训练阶段则用reward向上更新
            if self.training and len(actions) == self.K:
                try:
                    reward = self.env.env_method(method_name="done_",actions=actions)
                except Exception as e:
                    print(e)
                self.executor.submit(self.update_tree, None, reward[0], history)  
                break
            # 测试阶段则用最后一个访问的state的vaLue向上更新
            if not self.training and len(actions) == self.K and node.w is not None:
                self.executor.submit(self.update_tree, None, node.w, history)
                break
            
            # game_over, v, _ = self.env.done(state)
            # if game_over:
            #     v = v * 2
            #     self.executor.submit(self.update_tree, None, v, history)
            #     break
            # logger.debug(f" tree shape = {self.tree}")
            # print(self.node_lock[state])
            with self.node_lock[state]:
                if state not in self.tree:
                    # Expand and Evaluate
                    self.tree[state].sum_n = 1
                    self.tree[state].legal_moves = get_legal_mov(state, actions, self.K, is_root_node)
                    # print(self.tree[state].legal_moves)
                    # self.tree[state].legal_moves = list(state)
                    self.tree[state].waiting = True
                    # print("11111111111")
                    # logger.debug(f"expand_and_evaluate {state}, sum_n = {self.tree[state].sum_n}, history = {history}")
                    # if is_root_node and real_hist:
                    #     print("--22-22-2")
                    #     self.expand_and_evaluate(obs, history, real_hist)
                    # else:
                        # print("2222222222")
                        #把obs传进神经网络进行policy的估计
                    self.expand_and_evaluate(obs, history)
                    break

                # if state in history[:-1]: # loop
                #     for i in range(len(history) - 1):
                #         prit("10101010101010")
                #         if history[i] == state:
                #             if senv.will_check_or_catch(state, history[i+1]):
                #                 self.executor.submit(self.update_tree, None, -1, history)
                #             elif senv.be_catched(state, history[i+1]):
                #                 self.executor.submit(self.update_tree, None, 1, history)
                #             else:
                #                 # logger.debug(f"loop -> loss, state = {state}, history = {history[:-1]}")
                #                 self.executor.submit(self.update_tree, None, 0, history)
                #             break
                #     break

                # Select
                # print("execute search2....")
                node = self.tree[state]
                if node.waiting:
                    # node.visit.append(history)
                    node.visit['hist'].append(history)
                    node.visit['obs'].append(obs)
                    node.visit['root'].append(is_root_node)
                    node.visit['actions'].append(actions)
                    # print("888888888")
                    # logger.debug(f"history = {state}")
                    # logger.debug(f"wait for prediction state = {state}")
                    break
                # print("9999999999")
                sel_action = self.select_action_q_and_u(state, is_root_node)
                # print(sel_action)
                # print(state)

                virtual_loss = self.config["virtual_loss"]
                self.tree[state].sum_n += 1
                # logger.debug(f"node = {state}, sum_n = {node.sum_n}")
                
                action_state = self.tree[state].a[sel_action]
                action_state.n += virtual_loss
                action_state.w -= virtual_loss
                action_state.q = action_state.w / action_state.n

                # logger.debug(f"apply virtual_loss = {virtual_loss}, as.n = {action_state.n}, w = {action_state.w}, q = {action_state.q}")
                # if action_state.next is None:
                history.append(sel_action)
                actions.append(sel_action)
                # state = self.env.step(state, sel_action)
                obs[:,:,sel_action] = 0
                state = tuple([x for x in state if x != sel_action])
                is_root_node = False
                # list(state).remove(sel_action)
                history.append(state)
                # logger.debug(f"step action {sel_action}, next = {action_state.next}")

    def select_action_q_and_u(self, state, is_root_node) -> str:
        '''
        Select an action with highest Q(s,a) + U(s,a)
        '''
        is_root_node = self.root_state == state
        # logger.debug(f"select_action_q_and_u for {state}, root = {is_root_node}")
        node = self.tree[state]
        legal_moves = node.legal_moves

        # push p, the prior probability to the edge (node.p), only consider legal moves
        if node.p is not None:
            all_p = 0
            for mov in legal_moves:
                # mov_p = node.p[self.move_lookup[mov]]
                # node.a[mov].p = mov_p
                # all_p += mov_p
                node.a[mov].p = node.p[mov-1]
                all_p += node.p[mov-1]
            # rearrange the distribution
            if all_p == 0:
                all_p = 1
            for mov in legal_moves:
                node.a[mov].p /= all_p
            # release the temp policy
            node.p = None

        # sqrt of sum(N(s, b); for all b)
        xx_ = np.sqrt(node.sum_n + 1)  

        e = self.config["noise_eps"]
        c_puct = self.config["c_puct"]
        dir_alpha = self.config["dirichlet_alpha"]

        best_score = -99999999
        best_action = None
        move_counts = len(legal_moves)

        for mov in legal_moves:
            # 遍历所有合法动作，取得最优动作
            if is_root_node and self.no_act and mov in self.no_act:
                # logger.debug(f"mov = {mov}, no_act = {self.no_act}, continue")
                continue
            action_state = node.a[mov]
            p_ = action_state.p
            if is_root_node:
                p_ = (1 - e) * p_ + e * np.random.dirichlet(dir_alpha * np.ones(move_counts))[0]
            # Q + U
            score = action_state.q + c_puct * p_ * xx_ / (1 + action_state.n)
            # if score > 0.1 and is_root_node:
            #     logger.debug(f"U+Q = {score:.2f}, move = {mov}, q = {action_state.q:.2f}")
            if action_state.q > (1 - 1e-7):
                best_action = mov
                break
            if score >= best_score:
                best_score = score
                best_action = mov

        if best_action == None:
            ipdb.set_trace()
            logger.error(f"Best action is None, legal_moves = {legal_moves}, best_score = {best_score}")
        # if is_root_node:
        #     logger.debug(f"selected action = {best_action}, with U + Q = {best_score}")
        return best_action

    def expand_and_evaluate(self, obs, history, real_hist=None):
        '''
        Evaluate the state, return its policy and value computed by neural network
        '''
        if self.use_history:
            if real_hist:
                # logger.debug(f"real history = {real_hist}")
                # state_planes = self.env.state_history_to_planes(state, real_hist)
                pass
            else:
                # logger.debug(f"history = {history}")
                # state_planes = self.env.state_history_to_planes(state, history)
                pass
        else:
            # print("33333333333")
            # state_planes = self.env.state_to_planes(obs)
            state_planes = obs
        with self.q_lock:
            # 先将q_lock上锁，只有一个线程能往里加数据
            # print("2222222222")
            self.buffer_planes.append(state_planes) #存在用于预测的obs
            self.buffer_history.append(history)
            # logger.debug(f"EAE append buffer_history history = {history}")

    def update_tree(self, p, v, history):
        state = history.pop()

        #if assign predicted value, e.g., line 138, the p is not None, only execute 344~354 line and end;
        #if update the tree value, e.g., line 207, p is None, will execute 357~375
        if p is not None:
            with self.node_lock[state]:
                # logger.debug(f"return from NN state = {state}, v = {v}")
                # print("6666666666")
                node = self.tree[state]
                node.p = p
                node.w = v
                node.waiting = False
                if self.debugging:
                    self.debug[state] = (p, v)
                # logger.debug(f"node visit history {node.visit}")
                # for hist in node.visit:
                for obs, hist, is_root, actions in zip(node.visit['obs'], node.visit['hist'], node.visit['root'], node.visit['actions']):
                    # print("77777777777")
                    self.executor.submit(self.MCTS_search, obs, state, hist, is_root, actions)
                # node.visit = []
                node.visit = {'hist':[], 'obs':[]}

        virtual_loss = self.config["virtual_loss"]
        # logger.debug(f"backup from {state}, v = {v}, history = {history}")
        # 对于更新树节点的神经网络预测值时，以下不执行
        while len(history) > 0:
            action = history.pop()
            state = history.pop()
            # v = - v
            # print("122121212")
            with self.node_lock[state]:
                node = self.tree[state]
                action_state = node.a[action]
                action_state.n += 1 - virtual_loss
                action_state.w += v + virtual_loss
                action_state.q = action_state.w * 1.0 / action_state.n
                # logger.debug(f"update value: state = {state}, action = {action}, n = {action_state.n}, w = {action_state.w}, q = {action_state.q}")
        # t_lock 线程锁，对线程计数(减1)时上锁
        with self.t_lock:
            self.num_task -= 1
            # logger.debug(f"finish 1, remain num task = {self.num_task}")
            if self.num_task <= 0:
                self.all_done.release()

    def calc_policy(self, state, no_act) -> np.ndarray:
        '''
        calculate π(a|s0) according to the visit count
        '''
        node = self.tree[state]
        policy = np.zeros(self.labels_n)
        # max_q_value = -100
        debug_result = {}

        for mov, action_state in node.a.items():
            # policy[self.move_lookup[mov]] = action_state.n
            policy[mov-1] = action_state.n
            if no_act and mov in no_act:
                policy[mov-1] = 0
                continue
            if self.debugging:
                debug_result[mov] = (action_state.n, action_state.q, action_state.p)
            # if action_state.q > max_q_value:
            #     max_q_value = action_state.q

        # if max_q_value < self.play_config.resign_threshold and self.enable_resign and turns > self.play_config.min_resign_turn:
        #     return policy, True

        # if self.debugging:
        #     temp = sorted(range(len(policy)), key=lambda k: policy[k], reverse=True)
        #     for i in range(5):
        #         index = temp[i]
        #         mov = ActionLabelsRed[index]
        #         if mov in debug_result:
        #             self.search_results[mov] = debug_result[mov]

        policy /= np.sum(policy)
        return policy

    # def print_depth_info(self, state, turns, start_time, value, no_act):
    #     '''
    #     info depth xx pv xxx
    #     '''
    #     depth = self.done_tasks // 100
    #     end_time = time()
    #     pv = ""
    #     i = 0
    #     while i < 20:
    #         node = self.tree[state]
    #         bestmove = None
    #         root = True
    #         n = 0
    #         if len(node.a) == 0:
    #             break
    #         for mov, action_state in node.a.items():
    #             if action_state.n >= n:
    #                 if root and no_act and mov in no_act:
    #                     continue
    #                 n = action_state.n
    #                 bestmove = mov
    #         if bestmove is None:
    #             logger.error(f"state = {state}, turns = {turns}, no_act = {no_act}, root = {root}, len(as) = {len(node.a)}")
    #             break
    #         state = self.env.step(state, bestmove)
    #         root = False
    #         if turns % 2 == 1:
    #             bestmove = flip_move(bestmove)
    #         bestmove = senv.to_uci_move(bestmove)
    #         pv += " " + bestmove
    #         i += 1
    #         turns += 1
    #     if state in self.debug:
    #         _, value = self.debug[state]
    #         if turns % 2 != self.side:
    #             value = -value
    #     score = int(value * 1000)
    #     duration = end_time - start_time
    #     nps = int(depth * 100 / duration) * 1000
    #     output = f"info depth {depth} score {score} time {int(duration * 1000)} pv" + pv + f" nps {nps}"
    #     print(output)
    #     logger.debug(output)
    #     sys.stdout.flush()
        

    def apply_temperature(self, policy, steps) -> np.ndarray:
        if self.training and steps < 100000 and self.config["tau_decay_rate"] != 0:
            tau = np.power(self.config["tau_decay_rate"], steps/100)
        else:
            tau = 0
        if tau < 0.1 or not self.training:
             tau = 0
        if self.increase_temp and self.training:
            tau = 0.5
        if tau == 0:
            action = np.argmax(policy)
            ret = np.zeros(self.labels_n)
            ret[action] = 1.0
            return ret
        else:
            ret = np.power(policy, 1 / tau)
            ret /= np.sum(ret)
            return ret


class PortfolioSelector:
    def __init__(self, env, config, pipes=None, pid=None, K=None, use_history=False):
        self.env = env
        self.config = config
        self.searcher = None
        self.cur_pipes = pipes
        self.id = pid
        self.buffer = []
        self.pid = os.getpid()
        self.use_history = use_history
        self.K = K

        # current_model, use_history = load_model(config)
        # m = Manager()
        # cur_pipes = m.list([current_model.get_pipes() for _ in range(config.play.max_processes)])
        # self.cur_pipes = cur_pipes

    # def start(self):
    #     self.pid = os.getpid()
    #     ran = self.config.play.max_processes if self.config.play.max_processes > 5 else self.config.play.max_processes * 2
    #     sleep((self.pid % ran) * 10)
    #     logger.debug(f"Selfplay#Start Process index = {self.id}, pid = {self.pid}")

    #     idx = 1
    #     self.buffer = []
    #     search_tree = defaultdict(VisitState)

    #     while True:
    #         start_time = time()
    #         search_tree = defaultdict(VisitState)
    #         value, turns, state, store = self.select_portfolio(idx, search_tree)
    #         end_time = time()
    #         logger.debug(f"Process {self.pid}-{self.id} play game {idx} time={(end_time - start_time):.1f} sec, "
    #                      f"turn={turns / 2}, winner = {value:.2f} (1 = red, -1 = black, 0 draw)")
    #         if turns <= 10:
    #             senv.render(state)
    #         if store:
    #             idx += 1
    #         sleep(random())

    def select_portfolio(self, observation, num_step, training=False):
        # 从蒙特卡洛树中搜索K个(深度)的动作
        # Input 输入: obs 当前状态特征，
        #             num_step step的次数，
        #             training属于训练阶段还是测试阶段(True:训练时simulation时使用reward更新;False:测试时Simulation用预测的value更新)
        # Output 输出： action with length K, state_value 动作，state的值
        pipes = self.cur_pipes.pop()          #为第个进程配置一个通信管道

        # if not self.config.play.share_mtcs_info_in_self_play or \
        #     idx % self.config.play.reset_mtcs_info_per_game == 0:
        #     search_tree = defaultdict(VisitState)
        search_tree = defaultdict(VisitState)

        # if random() > self.config.play.enable_resign_rate:
        #     enable_resign = True
        # else:
        #     enable_resign = False

        self.searcher = MCTS(self.env, self.config, search_tree=search_tree, pipes=pipes, 
                                K=self.K, debugging=False, use_history=self.use_history, training=training)

        # state = senv.INIT_STATE
        state = [i for i in range(1, self.env.action_space[0].n+1)]  # state表示搜索树的节点状态，指当前有哪些资产没被选择，而obs指这些没被选择资产的特征
        num_step = num_step
        # history = [state]
        policys = [] 
        value = 0
        turns = 0       # even == red; odd == black
        finish = False
        final_move = None
        no_eat_count = 0
        check = False
        no_act = []
        actions = []
        increase_temp = False
        obs = copy.deepcopy(observation)
        while not finish:  #当depth尝试达到K时，结束选择
            start_time = time()
            # action, policy = self.searcher.action(state, turns, no_act, increase_temp=increase_temp)
            action, pi = self.searcher.action(obs, tuple(state), num_step, actions, increase_temp=increase_temp)
            actions.append(action)
            policys.append(pi)
            obs[:,:,action]=0
            state.remove(action)
            state = [x for x in state if x != action]
            # if len(actions) == selected_num:
            #     value = env.done(state)
            # if action is None:
            #     logger.debug(f"{turns % 2} (0 = red; 1 = black) has resigned!")
            #     value = -1
            #     break
            # if self.config.opts.log_move:
            #     logger.info(f"Process{self.pid} Playing: {turns % 2}, action: {action}, time: {(end_time - start_time):.1f}s")
            # logger.info(f"Process{self.pid} Playing: {turns % 2}, action: {action}, time: {(end_time - start_time):.1f}s")
            # history.append(action)

            if len(actions) == self.K:
                finish = True
            end_time = time()

            # try:
            #     state, no_eat = senv.new_step(state, action)
            # except Exception as e:
            #     logger.error(f"{e}, no_act = {no_act}, policy = {policy}")
            #     game_over = True
            #     value = 0
            #     break
            # turns += 1
            # if no_eat:
            #     no_eat_count += 1
            # else:
            #     no_eat_count = 0
            # history.append(state)

            # if no_eat_count >= 120 or turns / 2 >= self.config.play.max_game_length:
            #     game_over = True
            #     value = 0
            # else:
            #     game_over, value, final_move, check = senv.done(state, need_check=True)
                # if not game_over:
                    # if not senv.has_attack_chessman(state):
                    #     logger.info(f"双方无进攻子力，作和。state = {state}")
                    #     game_over = True
                    #     value = 0
                # increase_temp = False
                # no_act = []
                # if not game_over and not check and state in history[:-1]:
                #     free_move = defaultdict(int)
                #     for i in range(len(history) - 1):
                #         if history[i] == state:
                #             if senv.will_check_or_catch(state, history[i+1]):
                #                 no_act.append(history[i + 1])
                #             elif not senv.be_catched(state, history[i+1]):
                #                 increase_temp = True
                #                 free_move[state] += 1
                #                 if free_move[state] >= 3:
                #                     # 同一棋盘状态出现三次作和棋处理
                #                     game_over = True
                #                     value = 0
                #                     logger.info("闲着循环三次，作和棋处理")
                #                     break

        # if final_move:
            # policy = self.build_policy(final_move, False)
            # history.append(final_move)
            # policys.append(policy)
            # state = senv.step(state, final_move)
            # turns += 1
            # value = -value
            # history.append(state)

        self.searcher.close()
        del search_tree
        del self.searcher
        gc.collect()
        # if turns % 2 == 1:  # balck turn
        #     value = -value

        # v = value
        # if turns < 10:
        #     if random() > 0.9:
        #         store = True
        #     else:
        #         store = False
        # else:
        #     store = True

        # if store:
        #     data = [history[0]]
        #     for i in range(turns):
        #         k = i * 2
        #         data.append([history[k + 1], value])
        #         value = -value
        #     self.save_play_data(idx, data)

        #进程结束，配置的通信管道归还
        self.cur_pipes.append(pipes)
        # self.remove_play_data()
        return np.array([actions]), policys
        # return v, turns, state, store

    # def save_play_data(self, idx, data):
    #     self.buffer += data

    #     if not idx % self.config.play_data.nb_game_in_file == 0:
    #         return

    #     rc = self.config.resource
    #     utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    #     bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
    #     game_id = bj_dt.strftime("%Y%m%d-%H%M%S.%f")
    #     filename = rc.play_data_filename_tmpl % game_id
    #     path = os.path.join(rc.play_data_dir, filename)
    #     logger.info(f"Process {self.pid} save play data to {path}")
    #     write_game_data_to_file(path, self.buffer)
        # if self.config.internet.distributed:
        #     upload_worker = Thread(target=self.upload_play_data, args=(path, filename), name="upload_worker")
        #     upload_worker.daemon = True
        #     upload_worker.start()
        # self.buffer = []

    # def upload_play_data(self, path, filename):
    #     digest = CChessModel.fetch_digest(self.config.resource.model_best_weight_path)
    #     data = {'digest': digest, 'username': self.config.internet.username, 'version': '2.4'}
    #     response = upload_file(self.config.internet.upload_url, path, filename, data, rm=False)
    #     if response is not None and response['status'] == 0:
    #         logger.info(f"Upload play data {filename} finished.")
    #     else:
    #         logger.error(f'Upload play data {filename} failed. {response.msg if response is not None else None}')

    # def remove_play_data(self):
    #     files = get_game_data_filenames(self.config.resource)
    #     if len(files) < self.config.play_data.max_file_num:
    #         return
    #     try:
    #         for i in range(len(files) - self.config.play_data.max_file_num):
    #             os.remove(files[i])
    #     except:
    #         pass

    # def build_policy(self, action, flip):
    #     labels_n = len(ActionLabelsRed)
    #     move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
    #     policy = np.zeros(labels_n)

    #     policy[move_lookup[action]] = 1

    #     if flip:
    #         policy = flip_policy(policy)
    #     return list(policy)

 # Program start here 程序主入口
# def start(config: Config):
#     # set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
#     current_model, use_history = load_model(config)
#     m = Manager()
#     cur_pipes = m.list([current_model.get_pipes() for _ in range(config.play.max_processes)])
#     # play_worker = SelfPlayWorker(config, cur_pipes, 0)
#     # play_worker.start()
#     with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
#         futures = []
#         for i in range(config.play.max_processes):
#             selector = PortfolioSelector(config, cur_pipes, i, use_history)
#             logger.debug(f"Initialize Selector{i}...")
#             futures.append(executor.submit(selector.start))
