import ray
import os
import gin
import argparse
from D2MAV_A.agent import Agent
from D2MAV_A.runner import Runner
from copy import deepcopy
import time
import platform
import json
import numpy as np

os.environ["PYTHONPATH"] = os.getcwd()

import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.FATAL)
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser()
parser.add_argument("--cluster", action="store_true")
parser.add_argument("--learn_action", action="store_true")
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()


@gin.configurable
class Driver:
    def __init__(
            self,
            cluster=False,
            run_name=None,
            scenario_file=None,
            config_file=None,
            num_workers=1,
            iterations=1000,
            simdt=1,
            max_steps=1024,
            speeds=[0, 0, 84],
            LOS=10,
            dGoal=100,
            maxRewardDistance=100,
            intruderThreshold=750,
            rewardBeta=0.001,
            rewardAlpha=0.1,
            speedChangePenalty=0.001,
            shieldPenalty = 0.1,
            rewardLOS=-1,
            stepPenalty=0,
            clearancePenalty=0.005,
            gamma = 0.001,
            alpha = 0,
            gui=False,
            non_coop_tag=0,
            weights_file=None,
            run_type='train',
            traffic_manager_active=True,
            d2mav_active=True,
            vls_active=True

    ):

        self.cluster = cluster
        self.run_name = run_name
        self.run_type = run_type
        self.num_workers = num_workers
        self.simdt = simdt
        self.iterations = iterations
        self.max_steps = max_steps
        self.speeds = speeds
        self.LOS = LOS
        self.dGoal = dGoal
        self.maxRewardDistance = maxRewardDistance
        self.intruderThreshold = intruderThreshold
        self.rewardBeta = rewardBeta
        self.rewardAlpha = rewardAlpha
        self.speedChangePenalty = speedChangePenalty
        self.shieldPenalty = shieldPenalty
        self.rewardLOS = rewardLOS
        self.stepPenalty = stepPenalty
        self.clearancePenalty = clearancePenalty
        self.gamma = gamma
        self.alpha = alpha
        self.scenario_file = scenario_file
        self.config_file = config_file
        self.weights_file = weights_file
        print("weights: ", self.weights_file)
        self.gui = gui
        self.action_dim = 9
        # Modified to account for altitude
        self.observation_dim = 8
        self.context_dim = 9
        self.agent = Agent()
        self.agent_template = deepcopy(self.agent)
        self.working_directory = os.getcwd()
        self.non_coop_tag = non_coop_tag
        self.traffic_manager_active = traffic_manager_active
        self.d2mav_active = d2mav_active
        self.vls_active = vls_active

        if self.traffic_manager_active:
            self.observation_dim += 2

        self.agent.initialize(tf, self.observation_dim, self.context_dim, self.action_dim)

        if self.run_name is None:
            path_results = "results"
            path_models = "models"
        else:
            path_results = f"results/{self.run_name}"
            path_models = f"models/{self.run_name}"

        os.makedirs(path_results, exist_ok=True)
        os.makedirs(path_models, exist_ok=True)

        self.path_models = path_models
        self.path_results = path_results

    def train(self):

        # Start simulations on actors
        workers = {
            i: Runner.remote(
                i,
                self.agent_template,
                scenario_file=self.scenario_file,
                config_file=self.config_file,
                working_directory=self.working_directory,
                max_steps=self.max_steps,
                simdt=self.simdt,
                speeds=self.speeds,
                LOS=self.LOS,
                dGoal=self.dGoal,
                maxRewardDistance=self.maxRewardDistance,
                intruderThreshold=self.intruderThreshold,
                rewardBeta=self.rewardBeta,
                rewardAlpha=self.rewardAlpha,
                speedChangePenalty=self.speedChangePenalty,
                rewardLOS=self.rewardLOS,
                shieldPenalty=self.shieldPenalty,
                stepPenalty=self.stepPenalty,
                clearancePenalty=self.clearancePenalty,
                gamma = self.gamma,
                alpha = self.alpha,
                gui=self.gui,
                non_coop_tag=self.non_coop_tag,
                traffic_manager_active=self.traffic_manager_active,
                d2mav_active=self.d2mav_active,
                vls_active=self.vls_active
            )
            for i in range(self.num_workers)
        }

        rewards = []
        total_nmacs = []
        total_LOS = []
        max_travel_times = []
        iteration_record = []
        total_transitions = 0
        best_reward = -np.inf
        if self.agent.equipped:
            if self.weights_file is not None:
                self.agent.model.save_weights(self.weights_file)

            weights = self.agent.model.get_weights()
        else:
            weights = []
        print("GUI: ", self.gui)
        runner_sims = [workers[agent_id].run_one_iteration.remote(weights) for agent_id in workers.keys()]
        scenario = 0
        metric_list = []
        for i in range(self.iterations):

            done_id, runner_sims = ray.wait(runner_sims, num_returns=self.num_workers)
            results = ray.get(done_id)
            # Uncomment this when running with trained model
            transitions, workers_to_remove = self.agent.update_weights(results)
            # transitions = 0

            if self.agent.equipped:
                weights = self.agent.model.get_weights()

            total_reward = []
            mean_total_reward = None
            nmacs = []
            total_ac = []
            LOS_total = 0
            shield_total = 0
            shield_total_intersect = 0 
            shield_total_route = 0
            max_noise_increase = 0
            avg_noise_increase = 0
            scenario_file = None
            for result in results:
                data = ray.get(result)

                try:
                    total_reward.append(float(np.sum(data[0]["raw_reward"])))
                except:
                    pass

                if data[0]['environment_done']:
                    nmacs.append(data[0]['nmacs'])
                    total_ac.append(data[0]['total_ac'])

                LOS_total += data[0]['los_events']
                shield_total += data[0]['shield_events']
                shield_total_intersect += data[0]['shield_events_i']
                shield_total_route += data[0]['shield_events_r']
                max_halt_time = data[0]['max_halting_time']
                max_noise_increase = data[0]['max_noise_increase']
                avg_noise_increase = data[0]['avg_noise_increase']
                max_travel_time = data[0]['max_travel_time']
                full_travel_times_temp = list(data[0]['full_travel_times'].values())
                scenario_file = data[0]['scenario_file']
                speed_change_counter = data[0]['speed_change_counter']
                alt_change_counter = data[0]['alt_change_counter']

            if total_reward:
                mean_total_reward = np.mean(total_reward)

            for j, nmac in enumerate(nmacs):
                scenario += 1
                print(f"     Scenario Complete     ")
                print("|------------------------------|")
                print(f"| Scenario File:      {scenario_file}      |")
                print(f"| Total NMACS:      {nmac}      |")
                print(f"| Total Aircraft:   {total_ac[j]}  |")
                roll_mean = np.mean(rewards[-150:])
                # print(f"| Raw Reward: {total_reward[-1:]}  |")
                print(f"| Rolling Mean Reward: {np.round(roll_mean, 1)}  |")
                print(f"| Max Travel Time: {max_travel_time}  |")
                print(f"| Number of LOS Events: {LOS_total}  |")
                print(f"| Maximum Noise Increase: {max_noise_increase}  |")
                print(f"| Average Noise Increase: {avg_noise_increase}  |")
                print(f"| Number of Shield Events: {shield_total}  |")
                print(f"| Number of Intersection Shield Events: {shield_total_intersect}  |")
                print(f"| Number of Route Shield Events: {shield_total_route}  |")
                print(f"| Number of Speed Changes: {speed_change_counter}  |")
                print(f"| Number of Altitude Changes: {alt_change_counter}  |")
                print("|------------------------------|")
                print(" ")
                metric_dict = {}
                metric_dict['scenario_num'] = scenario
                metric_dict['shield_total'] = shield_total
                metric_dict['shield_total_intersection'] = shield_total_intersect
                metric_dict['shield_total_route'] = shield_total_route
                metric_dict['max_travel_time'] = max_travel_time
                metric_dict['full_travel_times'] = full_travel_times_temp
                metric_dict['los'] = LOS_total
                metric_dict['scenario_name'] = scenario_file
                metric_dict['max_noise'] = max_noise_increase
                metric_dict['avg_noise'] = avg_noise_increase
                metric_dict['speed_change_counter'] = speed_change_counter
                metric_dict['alt_change_counter'] = alt_change_counter
                metric_list.append(metric_dict)
                total_nmacs.append(nmac)
                max_travel_times.append(max_travel_time)
                total_LOS.append(LOS_total)
                iteration_record.append(i)

            if mean_total_reward:
                rewards.append(mean_total_reward)
                np.save("{}/reward.npy".format(self.path_results), np.array(rewards))

            if len(nmacs) > 0:
                np.save("{}/nmacs.npy".format(self.path_results), np.array(total_nmacs))
                np.save("{}/iteration_record.npy".format(self.path_results), np.array(iteration_record))

            total_transitions += transitions

            if not mean_total_reward:
                mean_total_reward = 0

            # print(f"     Iteration {i} Complete     ")
            # print("|------------------------------|")
            # print(f"| Mean Total Reward:   {np.round(mean_total_reward, 1)}  |")
            # roll_mean = np.mean(rewards[-150:])
            # print(f"| Rolling Mean Reward: {np.round(roll_mean, 1)}  |")
            # print(f"| Number of LOS Events: {LOS_total}  |")
            # print(f"| Max Halting Time: {max_halt_time}  |")
            # print(f"| Number of Shield Events: {shield_total}  |")
            # print("|------------------------------|")
            # print(" ")

            if self.agent.equipped:
                if len(rewards) > 150:
                    if np.mean(rewards[-150:]) > best_reward:
                        best_reward = np.mean(rewards[-150:])
                        self.agent.model.save_weights("{}/best_model.h5".format(self.path_models))

                self.agent.model.save_weights("{}/model.h5".format(self.path_models))

            # for agent_id in workers_to_remove:
            #     workers[agent_id] = Runner.remote(
            #         agent_id,
            #         self.agent_template,
            #         scenario_file=self.scenario_file,
            #         config_file=self.config_file,
            #         working_directory=self.working_directory,
            #         max_steps=self.max_steps,
            #         simdt=self.simdt,
            #         speeds=self.speeds,
            #         LOS=self.LOS,
            #         dGoal=self.dGoal,
            #         maxRewardDistance=self.maxRewardDistance,
            #         intruderThreshold=self.intruderThreshold,
            #         rewardBeta=self.rewardBeta,
            #         rewardAlpha=self.rewardAlpha,
            #         speedChangePenalty=self.speedChangePenalty,
            #         rewardLOS=self.rewardLOS,
            #         stepPenalty=self.stepPenalty,
            #         gui=self.gui,
            #         non_coop_tag = self.non_coop_tag,
            #     )

            # if len(workers_to_remove) > 0:
            #     time.sleep(5)

            runner_sims = [workers[agent_id].run_one_iteration.remote(weights) for agent_id in workers.keys()]
        print("Mean Travel Times: ", np.mean(max_travel_times))
        print("Mean number of NMACS: ", np.mean(total_LOS))
        print(metric_list)
        with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/alt_mod/mod_alt_001_1.json', 'w') as file:
            json.dump(metric_list, file, indent=4)
            
    def evaluate(self):

        # Start simulations on actors
        workers = {
            i: Runner.remote(
                i,
                self.agent_template,
                scenario_file=self.scenario_file,
                config_file=self.config_file,
                working_directory=self.working_directory,
                max_steps=self.max_steps,
                simdt=self.simdt,
                speeds=self.speeds,
                LOS=self.LOS,
                dGoal=self.dGoal,
                maxRewardDistance=self.maxRewardDistance,
                intruderThreshold=self.intruderThreshold,
                rewardBeta=self.rewardBeta,
                rewardAlpha=self.rewardAlpha,
                speedChangePenalty=self.speedChangePenalty,
                rewardLOS=self.rewardLOS,
                stepPenalty=self.stepPenalty,
                # shieldPenalty=self.shieldPenalty 
                gui=self.gui,
                traffic_manager_active=self.traffic_manager_active
            )
            for i in range(self.num_workers)
        }

        rewards = []
        total_nmacs = []
        iteration_record = []
        total_transitions = 0
        best_reward = -np.inf
        max_travel_times = []
        iteration_record = []
        total_LOS = []
        
        if self.agent.equipped:
            self.agent.model.load_weights(self.weights_file)
            weights = self.agent.model.get_weights()
        else:
            weights = []

        runner_sims = [workers[agent_id].run_one_iteration.remote(weights) for agent_id in workers.keys()]
        metric_list = []
        scenario = 0
        for i in range(self.iterations):

            done_id, runner_sims = ray.wait(runner_sims, num_returns=self.num_workers)
            results = ray.get(done_id)

            total_reward = []

            nmacs = []
            total_ac = []
            LOS_total = 0
            shield_total = 0
            shield_total_intersect = 0 
            shield_total_route = 0
            scenario_file = None
            for result in results:
                scenario += 1
                data = ray.get(result)
                try:
                    total_reward.append(float(np.sum(data[0]["raw_reward"])))
                except:
                    pass
                # LOS_total += data[0]['los_counter']
                if data[0]['environment_done']:
                    nmacs.append(data[0]['nmacs'])
                    total_ac.append(data[0]['total_ac'])
                LOS_total += data[0]['los_events']
                shield_total += data[0]['shield_events']
                shield_total_intersect += data[0]['shield_events_i']
                shield_total_route += data[0]['shield_events_r']
                max_halt_time = data[0]['max_halting_time']
                max_noise_increase = data[0]['max_noise_increase']
                avg_noise_increase = data[0]['avg_noise_increase']
                max_travel_time = data[0]['max_travel_time']
                full_travel_times_temp = list(data[0]['full_travel_times'].values())
                scenario_file = data[0]['scenario_file']
                speed_change_counter = data[0]['speed_change_counter']
                alt_change_counter = data[0]['alt_change_counter']

            mean_total_reward = np.mean(total_reward)

            for j, nmac in enumerate(nmacs):
                print(f"     Scenario Complete     ")
                print("|------------------------------|")
                print(f"| Scenario File:      {scenario_file}      |")
                print(f"| Total NMACS:      {nmac}      |")
                print(f"| Total Aircraft:   {total_ac[j]}  |")
                # roll_mean = np.mean(rewards[-150:])
                # print(f"| Rolling Mean Reward: {np.round(roll_mean, 1)}  |")
                print(f"| Max Travel Time: {max_travel_time}  |")
                print(f"| Number of LOS Events: {LOS_total}  |")
                print(f"| Maximum Noise Increase: {max_noise_increase}  |")
                print(f"| Average Noise Increase: {avg_noise_increase}  |")
                print(f"| Number of Shield Events: {shield_total}  |")
                print(f"| Number of Intersection Shield Events: {shield_total_intersect}  |")
                print(f"| Number of Route Shield Events: {shield_total_route}  |")
                print(f"| Number of Speed Changes: {speed_change_counter}  |")
                print(f"| Number of Altitude Changes: {alt_change_counter}  |")
                print("|------------------------------|")
                print(" ")
                total_nmacs.append(nmac)
                iteration_record.append(i)
                metric_dict = {}
                metric_dict['scenario_num'] = scenario
                metric_dict['shield_total'] = shield_total
                metric_dict['shield_total_intersection'] = shield_total_intersect
                metric_dict['shield_total_route'] = shield_total_route
                metric_dict['max_travel_time'] = max_travel_time
                metric_dict['full_travel_times'] = full_travel_times_temp
                print("Full Travel Times Temp: ", full_travel_times_temp)
                metric_dict['los'] = LOS_total
                metric_dict['scenario_name'] = scenario_file
                metric_dict['max_noise'] = max_noise_increase
                metric_dict['avg_noise'] = avg_noise_increase
                metric_dict['speed_change_counter'] = speed_change_counter
                metric_dict['alt_change_counter'] = alt_change_counter
                metric_list.append(metric_dict)
                total_nmacs.append(nmac)
                max_travel_times.append(max_travel_time)
                total_LOS.append(LOS_total)
                iteration_record.append(i)

            rewards.append(mean_total_reward)
            np.save("{}/eval_reward.npy".format(self.path_results), np.array(rewards))

            if len(nmacs) > 0:
                np.save("{}/eval_nmacs.npy".format(self.path_results), np.array(total_nmacs))
                np.save("{}/eval_iteration_record.npy".format(self.path_results), np.array(iteration_record))

            # total_transitions += transitions

            # print(f"     Iteration {i} Complete     ")
            # print("|------------------------------|")
            # print(f"| Mean Total Reward:   {np.round(mean_total_reward, 1)}  |")
            # roll_mean = np.mean(rewards[-150:])
            # print(f"| Rolling Mean Reward: {np.round(roll_mean, 1)}  |")
            # print(f"| Number of LOS Events: {LOS_total}  |")
            # print("|------------------------------|")
            # print(" ")
            runner_sims = [workers[agent_id].run_one_iteration.remote(weights) for agent_id in workers.keys()]
        print("Mean Travel Times: ", np.mean(max_travel_times))
        print("Mean number of NMACS: ", np.mean(total_LOS))
        print(metric_list)
        with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/eval/aircraft_mod_alt_train_01_0.json', 'w') as file:
            json.dump(metric_list, file, indent=4)


### Main code execution
# Uncomment this for training
# gin.parse_config_file("conf/config.gin")
gin.parse_config_file("conf/config_demo.gin")

if args.cluster:
    ## Initialize Ray
    ray.init(address=os.environ["ip_head"])
    print(ray.cluster_resources())
else:
    # check if running on Mac
    if platform.release() == "Darwin":
        ray.init(_node_ip_address="0.0.0.0", local_mode=args.debug)
    else:
        ray.init(local_mode=args.debug)
    print(ray.cluster_resources())

# Now initialize the trainer with 30 workers and to run for 100k episodes 3334 episodes * 30 workers = ~100k episodes
Trainer = Driver(cluster=args.cluster)
if Trainer.run_type == 'train':
    Trainer.train()
else:
    Trainer.evaluate()
