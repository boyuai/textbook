# -*- coding: utf-8 -*-
import random
import cityflow
import numpy as np
import os 
from collections import deque

# 如果没有安装 keras 和 tensorflow 库
# 请使用 pip install keras tensorflow 安装
from keras.layers import Input, Dense, Multiply, Add, concatenate
from keras.models import Model, load_model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.engine import Layer

class State(object):
    """
    定义一个路口的交通状态
    """
    def __init__(self, num_of_vehicles, num_of_waiting_vehicles, cur_phase):
        # 12条车道上正在运行的车辆数
        self.num_of_vehicles = num_of_vehicles
        # 12条车道上等待的车辆数
        self.num_of_waiting_vehicles = num_of_waiting_vehicles
        # 当前信号灯状态。信号灯有9种可能的状态，我们用0-8的9个整数来表示
        self.cur_phase = cur_phase


class TrafficEnv:
    def __init__(self):
        """
        创建环境，配置道路环境信息。
        """
        # 使用2x2的“井”字形路网环境
        config = "data/2x2/config.json"
        # 创建CityFlow模拟器
        self.eng = cityflow.Engine(config, thread_num=4)
        # 存储十字路口的当前红绿灯状态
        self.cur_phase = {
            'intersection_1_2': 0,
            'intersection_1_1': 0,
            'intersection_2_1': 0,
            'intersection_2_2': 0
        }
        # 声明各个十字路口联结的四条道路
        self.lanes_dict = {
            'intersection_1_2': [
                "road_2_2_2",
                "road_1_3_3",
                "road_0_2_0",
                "road_1_1_1"
            ],
            'intersection_1_1': [
                "road_1_2_3",
                "road_0_1_0",
                "road_1_0_1",
                "road_2_1_2"
            ],
            'intersection_2_1': [
                "road_1_1_0",
                "road_2_0_1",
                "road_3_1_2",
                "road_2_2_3"
            ],
            'intersection_2_2': [
                "road_2_1_1",
                "road_3_2_2",
                "road_2_3_3",
                "road_1_2_0"
            ]
        }

    def _get_sublanes(self, lanes):
        """
        获取道路对应的车道
        举例而言，道路road_1_2_3对应的三条车道为road_1_2_3_0, road_1_2_3_1, road_1_2_3_2
        参数说明：
            道路列表
        返回值说明：
            获取每条道路对应的三条车道，将它们存放在一个新列表中返回
        """
        sublanes = []
        for lane in lanes:
            for i in range(3):
                sublanes.append(lane + "_{0}".format(i))
        return sublanes

    def get_state(self, lane_list, cur_phase):
        """
        获取一个十字路口的交通状态
        参数说明：
            lane_list: 由通向该路口的4条道路的名字组成的列表
            cur_phase: 该路口当前的红绿灯状态
        返回值说明：
            返回一个State类的实例作为该路口的状态
        """
        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()

        num_of_vehicles = []
        num_of_waiting_vehicles = []
        # 遍历4条道路对应的12条车道
        for sublane in self._get_sublanes(lane_list):
            # 将12条车道对应的车辆数和等待车辆数存放在列表中
            num_of_vehicles.append(lane_vehicle_count[sublane])
            num_of_waiting_vehicles.append(lane_waiting_vehicle_count[sublane])

        # 创建该路口的状态实例
        return State(
            num_of_vehicles=num_of_vehicles,
            num_of_waiting_vehicles=num_of_waiting_vehicles,
            cur_phase=cur_phase
        )

    def get_observation(self):
        """
        调用get_state函数，分别获取对2x2环境中的每个路口的状态
        """
        state_dict = {}
        for intersection, lanes in self.lanes_dict.items():
            state_dict[intersection] = self.get_state(lanes, self.cur_phase[intersection])
        return state_dict

    def _record_traffic_state(self):
        return [self.eng.get_lane_waiting_vehicle_count(), self.eng.get_lane_vehicles()]

    def calculate_reward(self, lane_dict, last_state, cur_state):
        """
        对某十字路口，通过状态转移前后的交通状态差异来计算奖励
        参数说明：
            lane_dict: 由通向该路口的4条道路的名字组成的列表
        """
        reward = 0
        # 遍历每条车道sublane
        for sublane in self._get_sublanes(lane_dict):
            # 对当前状态等待的每艘车辆施加0.25惩罚
            reward += cur_state[0][sublane] * (-0.25)
            vehicle_leaving = 0
            for vehicle in last_state[1][sublane]:
                if not vehicle in cur_state[1][sublane]:
                    # 若出现在前一个状态的车辆未出现在当前状态中，则说明它已经通过路口，给予智能体1的奖励
                    vehicle_leaving += 1
            reward += vehicle_leaving * 1
        return reward

    def take_action(self, action_dict):
        """
        对环境施加已选的动作，计算随之带来的奖励
        参数说明：
            action_dict是智能体为四个路口选定的动作，它是一个字典，可以通过十字路口的名称来索引红绿灯状态的序号
        返回值说明：
            返回reward_dict，是智能体在四个路口分别获得的奖励，数据结构与action_dict相同
        """

        reward_dict = {}
        # 记录状态转移前交通状态
        last_state = self._record_traffic_state()

        # 设置信号灯状态
        for intersection, action in action_dict.items():
            self.eng.set_tl_phase(intersection, action)
            self.cur_phase[intersection] = action

        # 时间流逝，交通环境转移到下一步
        self.eng.next_step()
        # 记录状态转移后交通状态
        cur_state = self._record_traffic_state()

        # 为每个十字路口计算奖励
        for intersection in action_dict.keys():
            reward_dict[intersection] = self.calculate_reward(self.lanes_dict[intersection], last_state, cur_state)

        return reward_dict

    def get_time(self):
        return self.eng.get_current_time()

    def get_score(self):
        return self.eng.get_score()


class Selector(Layer):
    """
    状态选择层
    功能说明：
        若select与输入x相同，则输出1，否则输出0，其中select是创建该层时需要给定的参数
    """

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        super(Selector, self).build(input_shape)

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def compute_output_shape(self, input_shape):
        return input_shape


class DQNAgent:
    # 声明智能体的状态所需包含的数据特征的名称和维度
    feature_list = [
        ('num_of_vehicles', 12),
        ('num_of_waiting_vehicles', 12),
        ('cur_phase', 1)
    ]

    def __init__(self, num_phases):
        # 红绿灯可用的状态数是该智能体的可选参数，本实验中有9个状态
        self.num_phases = num_phases
        # 折损因子
        self.gamma = 0.9
        #
        self.training_start = 256
        # 创建Q值网络和目标网络，用q_bar_outdated作为同步计数器
        self.q_network = self.build_network()
        self.target_q_network = self.build_network()
        self.q_bar_outdated = 0
        # 初始化经验池
        self.memory = deque(maxlen=1024)
        self._update_target_model()

    def _update_target_model(self):
        self.target_q_network.set_weights(self.q_network.get_weights())

    def build_network(self):
        """
        使用Keras搭建Q值网络
        """
        # 为每个状态特征创建输入节点
        input_node_dict = {}
        for name, dimension in self.feature_list:
            input_node_dict[name] = Input(shape=[dimension])

        # 将智能体的输入状态的多种特征拼接为一长条向量
        flatten_feature_list = [input_node for input_node in input_node_dict.values()]
        flatten_feature = concatenate(flatten_feature_list, axis=1)
        # 通过一层全连接神经网络进行特征提取
        shared_dense = Dense(20, activation="sigmoid")(flatten_feature)
        selected_q_values_list = []
        for phase in range(self.num_phases):
            # 为每个当前信号灯状态分别构建独立的全连接层，将第一层全连接提取到的特征关联至输出动作的Q值
            separate_hidden = Dense(20, activation="sigmoid")(shared_dense)
            q_values = Dense(self.num_phases, activation='linear')(separate_hidden)
            # 信号灯状态选择层，Q值网络只会输出当前信号灯状态所对应的独立全连接层的输出
            # 当phase与输入节点cur_phase对应同一个信号灯状态时，selector为1，否则为0
            selector = Selector(phase)(input_node_dict['cur_phase'])
            # 同样的，当phase与cur_phase相同时，selected_q_values与输出Q值q_values相同，否则为0
            selected_q_values = Multiply()([q_values, selector])
            selected_q_values_list.append(selected_q_values)
        # 通过将选择后的Q值相加，来达到只选择当前状态对应的Q值的效果
        q_values = Add()(selected_q_values_list)

        network = Model(inputs=[input_node_dict[name] for name, _ in self.feature_list], outputs=q_values)
        network.compile(optimizer=RMSprop(lr=0.001), loss="mean_squared_error")

        return network

    def remember(self, state, action, reward, next_state):
        self.memory.append([state, action, reward, next_state])
        return

    def _sample_memory(self):
        sample_size = min(len(self.memory), 256)
        sampled_memory = random.sample(self.memory, sample_size)
        return sampled_memory

    def get_training_sample(self, memory_slices):
        # 将一批记忆(s,a,r,s')的展开为s,a,r,s'，将它们解压缩为若干列表，
        [state_list, action_list, reward_list, next_state_list] = list(zip(*memory_slices))
        # 将state_list与next_state_list中的每一个状态从State类型转换成列表类型
        state_input_list = [self.convert_state_to_input(state) for state in state_list]
        next_state_input_list = [self.convert_state_to_input(next_state) for next_state in next_state_list]
        # 再次进行解压缩操作， 所得到的state_input是一个有三个元素的列表，其中三个元素是是三个状态特征的列表，
        # 它们分别存放着车道车辆数、车道等待车辆数、当前信号灯状态
        state_input = list(zip(*state_input_list))
        next_state_input = list(zip(*next_state_input_list))
        # 将三个状态特征列表转换为numpy数组，以便作为输入供给Q值网络
        X = [np.array(state_feature) for state_feature in state_input]
        X_next = [np.array(state_feature) for state_feature in next_state_input]

        # 分别用目标网络和当前网络输出新老状态的Q值，用于计算目标Q值
        target = self.q_network.predict(X)
        bootstrap = self.target_q_network.predict(X_next)

        # 通过Deep Q-learning的更新公式，计算目标Q值，作为训练数据的标签
        for i in range(target.shape[0]):
            target[i][action_list[i]] = reward_list[i] + self.gamma * np.amax(bootstrap[i])
        Y = np.array(target)
        # 返回训练输入和标签
        return X, Y

    def _fit_network(self, X, Y):
        epochs = 3
        batch_size = min(32, len(Y))
        self.q_network.fit(X, Y, batch_size=batch_size, epochs=epochs, verbose=False)

    def train_network(self):
        """
        采样经验，计算目标Q值，训练Q值网络
        """
        # 当经验池容量过小时不进行训练
        if (len(self.memory) < self.training_start):
            return
        # 随机采样经验池中的历史交互数据(s,a,r,s')
        sampled_memory = self._sample_memory()
        # 计算目标Q值，作为Q值网络的训练标签
        # 对采样的数据进行格式转换。将一批训练数据(s,a)堆叠成X，并让标签Y与之对应，整理为训练数据X与训练标签Y的格式
        X, Y = self.get_training_sample(sampled_memory)
        # 增加目标网络的同步计数器
        self.q_bar_outdated += 1
        # 训练Q值网络
        self._fit_network(X, Y)

    def choose_action(self, state, greedy=False):
        """
        智能体根据状态选择动作
        """
        # 调整输入状态以符合Q值网络的输入格式
        state = self.convert_state_to_input(state)
        state = [np.reshape(state_feature, newshape=[1, -1]) for state_feature in state]
        # 通过Q值网络取得当前状态下各个动作的Q值
        q_values = self.q_network.predict(state)

        if (not greedy) and random.random() <= 0.01:
            # 使用epsilon-greedy策略下的随机动作选择
            action = random.randrange(len(q_values[0]))
        else:
            # 贪心策略，选择Q值最大的动作
            action = np.argmax(q_values[0])
        return action

    def convert_state_to_input(self, state):
        """
        提取状态内作为数据特征的类成员属性，组成一个numpy列表，以作为Q值网络的输入
        """
        return [np.array(getattr(state, name)) for name, _ in self.feature_list]

    def synchronize_target_network(self):
        """
        每20次训练将目标网络与Q值网络同步，并重置目标网络计数器
        """
        if self.q_bar_outdated >= 20:
            self._update_target_model()
            self.q_bar_outdated = 0

    def load_model(self, name):
        self.q_network = load_model("./model/dqn_{0}.h5".format(name), custom_objects={'Selector': Selector})

    def save_model(self, name):
        model_path = './model'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.q_network.save("./model/dqn_{0}.h5".format(name))


def intelli_traffic():
    # 创建智能体
    deeplight = DQNAgent(num_phases=9)
    num_epoch = 5
    num_step = 4096

    # 进行num_epoch轮训练
    for epoch in range(num_epoch):
        # 创建交通环境
        env = TrafficEnv()
        # 每次训练持续num_step个时间单位
        for i in range(num_step):
            # 获取四个十字路口的当前状态，存放在字典中
            state_dict = env.get_observation()

            # 智能体分别观察各个路口的状态，调控相应的信号灯设置
            action_dict = {}
            for intersection, state in state_dict.items():
                # 在前三轮使用epsilon greedy策略进行探索，之后使用greedy策略以求表现
                action = deeplight.choose_action(state, epoch > 3)
                action_dict[intersection] = action
             
            # 智能体做出动作，时间流逝一个单位，路口交通进行状态转移，并为智能体提供奖励
            reward_dict = env.take_action(action_dict)
            
            # 智能体观察下一步的状态
            next_state_dict = env.get_observation()

            # 将与环境交互的历史信息(s,a,r,s')存储在记忆池中
            for intersection in state_dict.keys():
                deeplight.remember(state_dict[intersection],
                                   action_dict[intersection],
                                   reward_dict[intersection],
                                   next_state_dict[intersection])

            # 智能体回顾记忆池中的信息，进行学习
            deeplight.train_network()
            # 每隔若干轮学习同步目标网络
            deeplight.synchronize_target_network()
            # 输出训练信息
            if i % 100 == 0:
                print("epoch: {}, time_step: {}, score {:.4f}".format(epoch, i, env.get_score()))
        # 保存模型 
        deeplight.save_model("model_{0}".format(epoch))

intelli_traffic()
