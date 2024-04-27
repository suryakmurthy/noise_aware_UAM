import ray
import os
import gin
import argparse
from D2MAV_A.agent import Agent
from D2MAV_A.runner import Runner
from bluesky.tools import geo
from copy import deepcopy
import pandas as pd
import random
import time
import platform
import json
import numpy as np
import logging

class Communication_Node():
    import tensorflow as tf
    import bluesky as bs
    def __init__(self, scenario_file):
        self.bs.init(mode="sim", configfile=f'/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/settings.cfg')
        self.bs.net.connect()
        self.reset(scenario_file)
    def reset(self, scenario_file):
        self.bs.stack.stack(r'IC ' + scenario_file)
        self.bs.stack.stack("FF")
        self.bs.sim.step()  # bs.sim.simt = 0.0 AFTER the call to bs.sim.step()
        self.bs.stack.stack("FF")
    def send_command(self, cmd):
        self.bs.stack.stack(cmd)
        self.bs.net.update()
    def update(self):
        self.bs.sim.step()
        self.bs.net.update()


def generate_scenario(path, num_scenarios, num_aircraft, dep_interval):
    for n_s in range(0, num_scenarios):
        print(path + f"/aircraft_"+str(num_aircraft)+"/test_case_"+str(n_s)+".scn")
        folder_path = path + f"/aircraft_"+str(num_aircraft)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        f = open(path + f"/aircraft_"+str(num_aircraft)+"/test_case_"+str(n_s)+".scn", "w")

        f.write("00:00:00.00>TRAILS ON \n")
        f.write("\n")
        f.write("00:00:00.00>PAN 32.77173371	-96.83678249 \n")
        f.write("\n")

        # Load Route Names for Generation
        waypoint_data = pd.read_csv(
            f'/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/bluesky/resources/navdata/nav.dat',
            delimiter='\t')
        waypoint_names = waypoint_data.iloc[:, 7]
        route_waypoints = {}
        for i in waypoint_names:
            route_id = i[0:4]
            if route_id not in route_waypoints.keys():
                route_waypoints[route_id] = []
            route_waypoints[route_id].append(i)

        for i in range(num_aircraft):
            # plane="A"+str(i)
            route_id = random.choice(list(route_waypoints.keys()))
            while "NHW" in route_id:
                route_id = random.choice(list(route_waypoints.keys()))
            route_length = len(route_waypoints[route_id])
            plane = "P" + route_id + str(i)
            time = "00:00:" + str(i * dep_interval) + ".00"
            f.write(time + ">CRE " + plane + ",Mavic," + route_id + "1,0,0" + "\n")
            f.write(time + ">ORIG " + plane + " " + route_waypoints[route_id][0] + "\n")
            f.write(time + ">DEST " + plane + " " + route_waypoints[route_id][-1] + "\n")
            f.write(time + ">SPD " + plane + " 30" + "\n")
            f.write(time + ">ALT " + plane + " 800" + "\n")
            for wpt in route_waypoints[route_id]:
                f.write(time + ">ADDWPT " + plane + " " + wpt + " 400 40" + "\n")
            f.write(time + ">" + plane + " VNAV on \n")
            f.write("\n")

        f.close()

def generate_scenario_austin(out_path, demand_dict_path, route_dict_path, dep_interval):
    print(out_path)
    folder_path = out_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    f = open(out_path + f"/austin_env_full_ver_4.scn", "w")

    f.write("00:00:00.00>TRAILS ON \n")
    f.write("\n")
    f.write("00:00:00.00>PAN 30.29828195311632 -97.92645392342473 \n")
    f.write("\n")

    # Load Route Names for Generation
    with open(demand_dict_path, 'r') as file_1:
        demand_dict = json.load(file_1)
    with open(route_dict_path, 'r') as file_2:
        route_dict = json.load(file_2)
    total_aircraft_num = 0
    for route_name in route_dict.keys():
        # plane="A"+str(i)
        for aircraft_num in range(0, demand_dict[route_name]):
            plane = "P" + route_name + str(aircraft_num)
            time = "00:00:" + str(aircraft_num * dep_interval) + ".00"
            # print(route_dict[route_name])
            first_wpt = route_dict[route_name][0] + route_dict[route_name][1] + "1"
            last_wpt = route_dict[route_name][len(route_dict[route_name]) - 2] + route_dict[route_name][len(route_dict[route_name]) - 1] + "2"
            f.write(time + ">CRE " + plane + ",Mavic," + first_wpt + ",0,0" + "\n")
            f.write(time + ">ORIG " + plane + " " + first_wpt + "\n")
            f.write(time + ">DEST " + plane + " " + last_wpt + "\n")
            f.write(time + ">SPD " + plane + " 40" + "\n")
            f.write(time + ">ALT " + plane + " 800" + "\n")
            for index in range(0, len(route_dict[route_name]) - 1):
                waypoint_1 = route_dict[route_name][index] + route_dict[route_name][index+1] + "1"
                waypoint_2 = route_dict[route_name][index] + route_dict[route_name][index+1] + "2"
                f.write(time + ">ADDWPT " + plane + " " + waypoint_1 + " 800 40" + "\n")
                f.write(time + ">ADDWPT " + plane + " " + waypoint_2 + " 800 40" + "\n")
            f.write(time + ">" + plane + " VNAV on \n")
            f.write("\n")
            total_aircraft_num += 1

    f.close()

# Test Scenario Generation:
intervals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
generate_scenario_austin(f'/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/scenarios/generated_scenarios_austin', '/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/D2MAV_A/route_demand_dict.json', '/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/D2MAV_A/route_info_dict_ver_2.json', 20)

# scenario_file = f'C:\Users\surya\PycharmProjects\ISMS_39\ILASMS_func3a-update-routes\scenarios\generated_scenarios\test_case_0.scn'
# # scenario_file = r'C:\Users\surya\PycharmProjects\ISMS_39\ILASMS_func3a-update-routes\scenarios\basic_env.scn'
# # https://github.com/TUDelft-CNS-ATM/bluesky/wiki/navdb
# node_1 = Communication_Node(scenario_file)

# interval_1 = 1000
# interval_2 = 100000
# counter = 0
# counter_2 = 0
# counter_3 = 0
# # Simulation Update Loop: reset and load a new scenario once all vehicles have exited the simulation.
# while 1:
#     # time.sleep(0.01)
#     node_1.update()
























# counter += 1
#     if counter % interval_1 == 0:
#         counter_2 +=1
#         if counter_2 % 2 == 0:
#             print("setting speed to 20")
#             for id in node_1.bs.traf.id:
#                 node_1.send_command(r'SPD ' + id + ' 20')
#                 # node_1.send_command(r'PAN '+ id)
#         else:
#             print("setting speed to 30")
#             for id in node_1.bs.traf.id:
#                 node_1.send_command(r'SPD ' + id + ' 30')
#                 # node_1.send_command(r'PAN ' + id)