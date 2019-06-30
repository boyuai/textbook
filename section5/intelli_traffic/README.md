# CityFlow是为多智体强化学习设计的大规模城市交通模拟器

## 特色功能
- 细致地模拟每辆车的行为，还原最逼真的交通细节。 
- 支持自定义的道路网和车流配置。
- 为强化学习提供清晰的python接口。
- 精心设计的数据结构与算法加上多线程编程带来了迅捷的速度，能够运行城市规模的交通模拟。

以下提供了关于“第五部分：红绿灯的调度”中模拟器的必要内容，若想了解更多信息，请访问CityFlow的官方文档[official documents](https://cityflow.readthedocs.io/en/latest/index.html)。

# 安装指引
本仓库提供了“红绿灯调度”任务专用的CityFlow版本以及代码实现中的道路车流配置文件，读者可以从源代码安装该CityFlow。
**目前，我们只支持unix系统下的运行。*- 本教程基于Ubuntu 16.04。

1. 请使用 python 3.6 进行安装。我们不保证支持低于 3.6 的 python 版本。
2. 安装 cpp 依赖

    ```bash
    apt update && apt-get install -y build-essential libboost-all-dev cmake
    ```

3. 从 github 克隆本仓库

    ```bash
    git clone https://github.com/boyuai/textbook
    ```

4. 进入本仓库的根目录并运行

    ```bash
    cd section5/intelli_traffic/CityFlow
    pip3 install .
    ```

5. 等待安装完成后，CityFlow 便可以成功运行

    ```python
    import cityflow
    eng = cityflow.Engine
    ```

# 使用说明
## 创建引擎

``` bash
import cityflow
eng = cityflow.Engine(config_path, thread_num=1)
```

- ```config_path```: 道路配置文件路径
- ```thread_num```: 工作线程数量

下面给出了道路配置文件样例，各参数的含义请查阅官方文档 [docs](https://cityflow.readthedocs.io/en/latest/start.html).

```json
{
    "interval": 1.0,
    "warning": true,
    "seed": 0,
    "dir": "data/",
    "roadnetFile": "roadnet/testcase_roadnet_3x3.json",
    "flowFile": "flow/testcase_flow_3x3.json",
    "rlTrafficLight": false,
    "laneChange": false,
    "saveReplay": true,
    "roadnetLogFile": "frontend/web/testcase_roadnet_3x3.json",
    "replayLogFile": "frontend/web/testcase_replay_3x3.txt"
}
```

## 模拟运行

调用模拟器 ```eng.next_step()``` 计算一个时间单位后的交通状态。

```python
eng.next_step()
```

## 交通数据访问 API

```get_vehicle_count()```:

- 返回正在运行的车辆数。

```get_lane_vehicle_count()```:

- 返回每条车道上正在运行的车辆数。

```get_lane_waiting_vehicle_count()```:

- 返回每条车道上正在等待的车辆数。速度小于 0.1m/s 的车辆被视作等待。

```get_lane_waiting_vehicle_count()```:

- 返回每条道路上的车辆id。

```get_vehicle_speed()```:

- 返回每辆车的速度。

```get_vehicle_distance()```:

- 返回每辆车在车道上的已运行的距离。

```get_current_time()```:

- 返回模拟器时间(秒)。

```get_score()```:

- 返回车辆平均运行时间(秒)，作为衡量调度算法性能的指标。

## 红绿灯调控 API

```set_tl_phase(intersection_id, phase_id)```:

- 将id为 ```intersection_id``` 的红绿灯状态设置为 ```phase_id```，仅当 ```rlTrafficLight``` 被设置为 ```true``` 时生效。
- 可以在 ```roadnetFile``` 中查看红绿灯id以及可选状态。

## 可视化

代码运行完成后，读者可以使用网页前端来查看模拟器中的车辆运行情况。

1. 安装轻量级 Web 应用框架 —— flask。

```
pip3 install flask
```

2. 运行 flask

```bash
cd fontend/pixi
python3 app.py
```

并在浏览器中访问网址 ```http://localhost:8080/?roadnetFile=roadnet.json&logFile=replay.txt```。
