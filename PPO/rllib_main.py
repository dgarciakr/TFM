import matplotlib
matplotlib.use('Agg')
import ray
import numpy as np
import os
import json
import torch

from ray.rllib.agents.ppo           import PPOTrainer, DEFAULT_CONFIG
import matplotlib.pyplot as plt

def plot_rewards (rewards, dir):

    # Figure
    fig, ax = plt.subplots()
    plt.axis('on')

    rw = np.array(rewards)
    maxX = len(rw)

    # X_axis = np.linspace(1, X_AXIS, chunk)
    X_axis = np.arange(maxX)

    ax.set_xticks(X_axis)

    ax.set_xlabel('# Episode')
    ax.set_ylabel('Reward')
    ax.set_title("Reward per Episode")

    ax.grid()

    # Plot:
    ax.plot(X_axis, rw)

    #plt.show()
    plt.savefig(dir)
    return


def on_postprocess_traj(info):
    print("Agt_id", info["agent_id"])
    print("Eps_id", info["episode"].episode_id)
    print("Policy_obj ", info["pre_batch"][0])
    print("Sample_obj ", info["pre_batch"][1])

#####  MAIN  #####


for i in range(2):

    ray.init(num_gpus = torch.cuda.device_count(), num_cpus= 1)

    config = DEFAULT_CONFIG.copy()
    config["num_gpus"] = 1
    config["num_gpus_per_worker"] = torch.cuda.device_count()
    config["num_workers"] = 1     # No paralellism
    config["use_critic"] = True
    config["use_gae"] = True
    
    config["model"]["fcnet_hiddens"]= [32, 128, 256, 256, 128, 32]

    config["framework"] = "torch"     # tf2 works.
    config["log_level"] = "WARN"   # INFO (default) is too verbose
    config["gamma"] = 0.95         # Default Gamma
    config["lr"] = 0.0001          # The default learning rate.
    config["render_env"] = True

    directory_graphs = "graphs"
    directory_outputs = "data_output"

    for filename in os.listdir(directory_graphs):
        f = os.path.join(directory_graphs, filename)
        P = -1
        if os.path.isfile(f):

            filename = filename.split(".")[0]

            try:
                os.mkdir(directory_outputs +"/"+ filename)
            except OSError:
                print("La carpeta existe")

            output_file = os.path.join(directory_outputs, filename +"/")

            ep = 0

            with open(f, 'r') as fl:
                data = json.load(fl)
                P = data["Graph"]["P"]
                data["Output"]["output_file"] = str(output_file + filename +  "_" + str(i) + ".txt")
                ep = data["Hyperparameters"]["n_episodes"]
            fl.close()
            os.remove(f)

            with open(f, "w") as fl:
                json.dump(data, fl, indent = 4)
            fl.close()

            config["env_config"] = {"config_file": f}
            
            if i == 0:
                with open(f, 'r') as fl:
                    data = json.load(fl)
                    data["Config"]["reward_type"] = "num_msgs"
                    fl.close()
                    os.remove(f)
                
                with open(f, "w") as fl:
                    json.dump(data, fl, indent = 4)
                fl.close()

                agent = PPOTrainer(config = config, env="gym_hpc:Mapping-v1")

            elif i == 1:
                with open(f, 'r') as fl:
                    data = json.load(fl)
                    data["Config"]["reward_type"] = "volume"
                    fl.close()
                    os.remove(f)

                with open(f, "w") as fl:
                    json.dump(data, fl, indent = 4)
                fl.close()

                agent = PPOTrainer(config = config, env="gym_hpc:Mapping-v1")
            
            if i == 0:
                with open(f, 'r') as fl:
                    data = json.load(fl)
                    data["Config"]["reward_type"] = "num_msgs"
                    fl.close()
                    os.remove(f)

                with open(f, "w") as fl:
                    json.dump(data, fl, indent = 4)
                fl.close()

                agent = PPOTrainer(config = config, env='gym_hpc:Mapping-v1')
            
            elif i == 1:
                with open(f, 'r') as fl:
                    data = json.load(fl)
                    data["Config"]["reward_type"] = "volume"
                    fl.close()
                    os.remove(f)

                with open(f, "w") as fl:
                    json.dump(data, fl, indent = 4)
                fl.close()

                agent = PPOTrainer(config = config, env="gym_hpc:Mapping-v1")
                print("AGENT_STATE: ", agent.get_state)

            total_rewards = []

            for j in range(50):
                result = agent.train()
                print("Mean reward: %d -> %.3f" % (j+1, result["episode_reward_mean"]))
                total_rewards.append(result["episode_reward_mean"])
                print("Best reward: " , result["episode_reward_max"])
                print("Mean reward: " , result["episode_reward_mean"])
                print("Worst reward: " , result["episode_reward_min"])

            dir = output_file + "/" + filename + "_" + str(i) + ".png"

            plot_rewards(total_rewards, dir)

            ray.shutdown()