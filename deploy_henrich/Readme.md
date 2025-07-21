# Heinrich Deployment
This code is a modified version of Unitree's deployment code here: https://github.com/unitreerobotics/unitree_rl_gym/tree/main/deploy

It contains code for Sim2Sim (IsaacGym to MuJoCo) and Sim2Real testing.

## Sim2Sim
Sim2Sim means loading the policy trained in IsaacGym and deploying on MuJoCo. The 'sim2sim.py' script records the behavior of this agent, but doesn't assign rewards - it's not a full alternative environment, just a change in simulation backend. Just as for real deployment, success should be measured by behavior. 

## Sim2Real
Since this means working with the physical robot, make sure you've read the documentation and get certified for robot use. Prepare the physical robot before starting the sortware processes here and note the network interface.

Step 1: Check the configuration file. This contains the location of the policy you want to deploy among other things. Verify everything is in order.

Step 2: Start the deployment command as 'python sim2sim.py <network_interface> <config_name>'. This will set the robot joints to zero torque. Make sure this has happened and that Heinrich is now ready for action. Then move to an observation point.

Step 3: Press 'start' on the remote to begin deployment mode. Now the robot resets to default position. The 'A' button starts movement. Exit by pressing 'select' or 'ctrl+c' on your keyboard.