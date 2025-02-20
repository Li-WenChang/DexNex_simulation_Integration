import copy
import os
import time

import pydot
from IPython.display import SVG, display

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from pydrake.geometry import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
    RenderLabel,
    Role,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import BodyIndex
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdSensor,
)
from pydrake.visualization import (
    AddDefaultVisualization,
    ColorizeDepthImage,
    ColorizeLabelImage,
)

from pydrake.systems.primitives import ConstantVectorSource
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.controllers import InverseDynamics
from pydrake.geometry import SceneGraph
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.all import  ModelInstanceIndex
from pydrake.systems.primitives import SparseMatrixGain
from pydrake.systems.controllers import PidController
from pydrake.systems.primitives import PassThrough
from pydrake.systems.primitives import StateInterpolatorWithDiscreteDerivative
import csv
#---------------------------------------------------
import drake_ros.core
from drake_ros.core import ClockSystem
from drake_ros.core import RosInterfaceSystem
from sensor_msgs.msg import Image
import rclpy
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy

import helper
import kpid
#---------------------------------------------------
meshcat = StartMeshcat()

table_top_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="table_top">
    <link name="table_top_link">
      <visual name="visual">
        <pose>0 0 0.445 0 0 0</pose>
        <geometry>
          <box>
            <size>0.55 1.1 0.05</size>
          </box>
        </geometry>
        <material>
         <diffuse>0.9 0.8 0.7 1.0</diffuse>
        </material>
      </visual>
      <collision name="collision">
        <pose>0 0 0.445  0 0 0</pose>
        <geometry>
          <box>
            <size>0.55 1.1 0.05</size>
          </box>
        </geometry>
      </collision>
    </link>
    <frame name="table_top_center">
      <pose relative_to="table_top_link">0 0 0.47 0 0 0</pose>
    </frame>
  </model>
</sdf>


"""
cylinder_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="cylinder">
    <pose>0 0 0 0 0 0</pose>
    <link name="cylinder_link">
      <inertial>
        <mass>0.01</mass>
        <inertia>
          <ixx>0.005833</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.005833</iyy>
          <iyz>0.0</iyz>
          <izz>0.005</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.1</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
        <material>
          <diffuse>1.0 1.0 1.0 1.0</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
def xyz_rpy_deg(xyz, rpy_deg):
    """Shorthand for defining a pose."""
    rpy_deg = np.asarray(rpy_deg)
    return RigidTransform(RollPitchYaw(rpy_deg * np.pi / 180), xyz)

def read_csv_column(column_index):
    """
    Reads a specified column from 'dexnex_position_test.csv' and returns it as a list.
    
    :param column_index: Index of the column to read (0-based indexing)
    :return: List containing values from the specified column
    """
    column_list = []
    
    with open('drake_position_test.csv', mode='r', newline='') as file:
        reader = csv.reader(file)
        
        for row in reader:
            if row and len(row) > column_index:  # Ensure row is not empty and has the column
                column_list.append(row[column_index])
    
    return column_list

builder = DiagramBuilder()
#----------------------------------------------------------------------------------------------------
drake_ros.core.init()
rclpy.init()

interface = "images"
sys_ros_interface = builder.AddSystem(RosInterfaceSystem(interface))
ClockSystem.AddToBuilder(builder, sys_ros_interface.get_ros_interface())
#----------------------------------------------------------------------------------------------------
position_command_time_step = 0.1
simulation_plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
simulation_plant.set_name("plant")



# Parse the avatar into the simulation plant
dexnex_path1 = "/home/li-wen/avatar_ws_backup/learning_drake/avatar_new.urdf"
parser_sim = Parser(simulation_plant, "sim")
parser_sim.package_map().PopulateFromEnvironment('AMENT_PREFIX_PATH')
(dexnex_sim,) = parser_sim.AddModels(dexnex_path1)

# Parse the objects into the simulation plant
(table,) = parser_sim.AddModelsFromString(table_top_sdf, "sdf")
(cylinder,) = parser_sim.AddModelsFromString(cylinder_sdf, "sdf")

# fix table to the world frame
X_WorldTable = RigidTransform(RollPitchYaw(np.asarray([0, 0, 90]) * np.pi / 180), p=[0.5,1.5,0])
table_frame = simulation_plant.GetFrameByName("table_top_center")
simulation_plant.WeldFrames(simulation_plant.world_frame(), table_frame, X_WorldTable)


#---------------------------------------------------------------------------------------------
# adding camera to simulation_plant

renderer_name = "renderer"
scene_graph.AddRenderer(
    renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

# N.B. These properties are chosen arbitrarily.
intrinsics = CameraInfo(
    width=640,
    height=480,
    fov_y=np.pi/2,
)
core = RenderCameraCore(
    renderer_name,
    intrinsics,
    ClippingRange(0.01, 10.0),
    RigidTransform(),
)
color_camera = ColorRenderCamera(core, show_window=False)
depth_camera = DepthRenderCamera(core, DepthRange(0.01, 10.0))

world_id = simulation_plant.GetBodyFrameIdOrThrow(
                  simulation_plant.GetBodyByName("neck_cam_left_eye").index())
X_WB = xyz_rpy_deg([0, 0, 0.05], [0, 90, 180]) # change xyz later not rpy
sensor = RgbdSensor(
    world_id,
    X_PB=X_WB,
    color_camera=color_camera,
    depth_camera=depth_camera,
)

builder.AddSystem(sensor)
builder.Connect(
    scene_graph.get_query_output_port(),
    sensor.query_object_input_port(),
)

#------------------------------------------------------------------------------------------
simulation_plant.Finalize()

# Create the controller plant (only the robot)
controller_plant = builder.AddSystem(MultibodyPlant(time_step=0.001))
#controller_plant.RegisterAsSourceForSceneGraph(SceneGraph())  # Optional, for visualization
controller_plant.set_name("controller plant")

# Parse the robot into the controller plant
parser_control = Parser(controller_plant, "control")
parser_control.package_map().PopulateFromEnvironment('AMENT_PREFIX_PATH')
(dexnex_control,) = parser_control.AddModels(dexnex_path1)


# Finalize the controller plant
controller_plant.Finalize()

# printing out the model index in simulation plant
for i in range(simulation_plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = simulation_plant.GetModelInstanceName(model_instance)
        print(model_instance_name, "is", model_instance)

avatar_model = ModelInstanceIndex(2)
table = ModelInstanceIndex(3)
cylinder = ModelInstanceIndex(4)
#----------------------------------------------------------------------------------------------------
# set the configuration of table
cylinder = simulation_plant.GetBodyByName("cylinder_link")
X_WorldCylinder = RigidTransform(RollPitchYaw(np.asarray([0, 0, 90]) * np.pi / 180), p=[0.5,1.5,0.5])
simulation_plant.SetDefaultFreeBodyPose(cylinder, X_WorldCylinder)

#----------------------------------------------------------------------------------------------------
# set up a passthrough to get position input
num_dexnex_positions = simulation_plant.num_positions(avatar_model)
dexnex_position = builder.AddSystem(PassThrough(num_dexnex_positions))

builder.ExportInput(
    dexnex_position.get_input_port(),
    "dexnex" + "_position",
)

#----------------------------------------------------------------------------------------------------
# set up a system to convert position to position + velocity
desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_dexnex_positions,
                    position_command_time_step,
                    suppress_initial_transient=True,
                )
            )
builder.Connect(dexnex_position.get_output_port(), desired_state_from_position.get_input_port(),)
#----------------------------------------------------------------------------------------------------
# setting up PID controller
U = controller_plant.num_positions()
pid = PidController(kpid.Kp, kpid.Ki, kpid.Kd)
PID = builder.AddSystem(pid)

builder.Connect(simulation_plant.get_state_output_port(avatar_model),
                PID.get_input_port_estimated_state())


builder.Connect(desired_state_from_position.get_output_port(),
                PID.get_input_port_desired_state())
#----------------------------------------------------------------------------------------------------
# setting ID control
context = controller_plant.CreateDefaultContext()

id_system = InverseDynamics(controller_plant, InverseDynamics.kInverseDynamics, context)
ID = builder.AddSystem(id_system)
ID.set_name("Inverse Dynamics")

builder.Connect(simulation_plant.get_state_output_port(avatar_model),
                ID.get_input_port_estimated_state())

builder.Connect(PID.get_output_port_control(),
                ID.get_input_port_desired_acceleration())
               
#----------------------------------------------------------------------------------------------------
# convert generalized force to actuation
# D = controller_plant.MakeActuationMatrixPseudoinverse()

# matrix_torque2actuation = builder.AddSystem(SparseMatrixGain(D))
# matrix_torque2actuation.set_name("PseudoInverse Actuation matrix")



# builder.Connect(ID.get_output_port_generalized_force(),
#                 matrix_torque2actuation.get_input_port())

# builder.Connect(matrix_torque2actuation.get_output_port(),
#                 simulation_plant.get_actuation_input_port(avatar_model))

#----------------------------------------------------------------------------------------------------
# convert generalized force to actuation
# cheating
cheating_matrix = np.zeros((72, 66))
cheating_matrix[:66, :66] = np.eye(66)

matrix_torque2actuation = builder.AddSystem(SparseMatrixGain(cheating_matrix))
matrix_torque2actuation.set_name("PseudoInverse Actuation matrix")

builder.Connect(ID.get_output_port_generalized_force(),
                matrix_torque2actuation.get_input_port())

builder.Connect(matrix_torque2actuation.get_output_port(),
                simulation_plant.get_applied_generalized_force_input_port())


#----------------------------------------------------------------------------------------------------
AddDefaultVisualization(builder=builder, meshcat=meshcat)
diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
diagram.SetDefaultContext(diagram_context)


# following two lines make you see all state name
# print("Number of states: ", plant.num_multibody_states())
# print(plant.GetStateNames(True))
#----------------------------------------------------------------------------------------------------
# setting the initial states of dexnex
plant_context = diagram.GetMutableSubsystemContext(simulation_plant, diagram_context)
qv = simulation_plant.GetPositionsAndVelocities(plant_context)


qv[0] = np.pi/2
qv[30+0] = -np.pi/2
qv[62] = -np.pi/2



simulation_plant.SetPositionsAndVelocities(plant_context, qv)
#--------------------------------------------------------------------------------------------
svg_data = pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[0].create_svg()

#Save the SVG to a file
with open("dexnex_PIDandCamera.svg", "wb") as f:
    f.write(svg_data)

print("SVG saved as output.svg")
#-----------------------------------------------------------------------------------------------
ros_node = rclpy.create_node('image_publisher_node')

qos = QoSProfile(
    depth=10,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE)

rgb_publisher = ros_node.create_publisher(Image, '/camera/rgb_image', 10)
depth_publisher = ros_node.create_publisher(Image, '/camera/depth_image', 10)

def publish_rgb_image(numpy_image, encoding="rgb8"):
    msg = Image()
    msg.height = numpy_image.shape[0]
    msg.width = numpy_image.shape[1]
    msg.encoding = encoding
    msg.step = numpy_image.shape[1] * numpy_image.shape[2]  # Bytes per row
    msg.data = numpy_image.tobytes()
    msg.header.stamp = ros_node.get_clock().now().to_msg()
    rgb_publisher.publish(msg)
def publish_depth_image(numpy_image, encoding="32FC1"):
    msg = Image()
    msg.height = numpy_image.shape[0]
    msg.width = numpy_image.shape[1]
    msg.encoding = encoding
    msg.step = numpy_image.shape[1] * numpy_image.shape[2]
    msg.data = numpy_image.tobytes()
    msg.header.stamp = ros_node.get_clock().now().to_msg()
    depth_publisher.publish(msg)
#-----------------------------------------------------------------------------------------------
# set up simulator
simulator = Simulator(diagram, diagram_context) # remember to add diagram_context to the argument
simulator.Initialize()
simulator.set_target_realtime_rate(1.0)
#-----------------------------------------------------------------------------------------------
# run the simulation
time.sleep(5)

finish_time = position_command_time_step*52
which_command = 0

while simulator.get_context().get_time() < finish_time:
    position_list = read_csv_column(which_command)
    simulator_context = simulator.get_mutable_context()
    
    # give position command to controller
    diagram.get_input_port(0).FixValue(simulator_context, position_list)

    # publish RGB and depth image
    color = sensor.color_image_output_port().Eval(sensor.GetMyContextFromRoot(simulator_context)).data
    rgb_image = color[:, :, :3]    
    depth = sensor.depth_image_32F_output_port().Eval(sensor.GetMyContextFromRoot(simulator_context)).data
    

    publish_rgb_image(rgb_image, encoding ="rgb8")
    publish_depth_image(depth, encoding = "32FC1")
    
    simulator.AdvanceTo(simulator.get_context().get_time() + position_command_time_step)

    # extend the simulation for 2 sec------------------------------------------------
    if which_command == 51:
      simulator.AdvanceTo(simulator.get_context().get_time() + 2.0)
    # -------------------------------------------------------------------------------
    
    which_command += 1

  
print("finished")