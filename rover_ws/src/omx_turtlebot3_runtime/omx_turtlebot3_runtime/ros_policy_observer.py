"""Real ROS encoders/RGB -> v6 observations for the C++ mp_control relay.

No motor I/O, model inference, simulated feedback, or commissioning flag changes.
Current is unavailable in the legacy JointState interface and is not a feature
of the 69D policy observation. This reference-relay trial keeps contact/deck
bits false; it does not claim measured grasp or placement confirmation.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import time
import math
import cv2
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String, Bool
from geometry_msgs.msg import PointStamped, PoseStamped
from scipy.spatial.transform import Rotation
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer
from .core import RobotSample, PoseCache, ObjectPose, Phase, observation
from .planning import Planner
from .vision import Detector, synchronized_joints, paired_camera_info


def status_phase(text):
    text = text.lower()
    if 'stay' in text and ('handoff' in text or 'returning' in text or 'cargo_loaded' in text):
        return Phase.RETURN_STAY
    if 'release' in text or 'opening gripper on follower' in text:
        return Phase.PLACE
    if 'handoff place' in text:
        return Phase.EXTEND_LOWER
    if 'joint1 turn' in text or 'handoff rotate' in text:
        return Phase.ROTATE_RIGHT
    if 'lift' in text and 'handoff' in text:
        return Phase.RETRACT
    if 'gripper close' in text or 'grasp settle' in text:
        return Phase.GRIP
    if 'pregrasp' in text or 'eef' in text:
        return Phase.REACH
    if 'grasp sequence started' in text:
        return Phase.OPEN
    return Phase.RECOGNIZE


class Observer(Node):
    def __init__(self, args):
        super().__init__('omx_v6_ros_policy_observer')
        self.args=args
        self.selected=int(getattr(args,'selected_id',0))
        if self.selected not in (0,1):raise ValueError('Unsupported marker ID')
        self.cfg=yaml.safe_load(args.config.read_text())
        self.training=yaml.safe_load(args.training.read_text())
        self.planner=Planner(args.scene,self.training,self.cfg)
        self.detector=Detector(self.cfg['vision']);self.bridge=CvBridge()
        self.history=deque(maxlen=500);self.cache=PoseCache()
        self.phase=Phase.RECOGNIZE;self.goal=None;self.previous=np.zeros(4)
        self.target_base=None
        self.planned=time.monotonic();self.done=False;self.count=0;self.saved=-math.inf
        args.log_dir.mkdir(parents=True,exist_ok=False)
        self.log=(args.log_dir/'observer.jsonl').open('x',buffering=1)
        self.obs=self.create_publisher(Float32MultiArray,'/omx_v6/observation',10)
        self.point=self.create_publisher(PointStamped,'/target/object_in_base',10)
        self.aruco_pose=self.create_publisher(PoseStamped,'/target/aruco_pose',10)
        self.bbox=self.create_publisher(Float32MultiArray,'/target/eef_tracked_bbox',10)
        self.front_bbox=self.create_publisher(Float32MultiArray,'/target/tracked_bbox',10)
        self.close_ready=self.create_publisher(Bool,'/target/close_range_ready',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.input_ready=self.create_publisher(Bool,'/omx_v6/observer_ready',QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(JointState,'/joint_states',self.joints,qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray,'/omx_v6/policy_goal',self.on_goal,10)
        self.create_subscription(Float64MultiArray,'/omx_v6/policy_action',self.on_action,10)
        self.create_subscription(String,'/mp_control/status',self.status,QoSProfile(depth=10,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.image=Subscriber(self,Image,'/eef_camera/image_raw',qos_profile=qos_profile_sensor_data)
        self.info=Subscriber(self,CameraInfo,'/eef_camera/camera_info',qos_profile=qos_profile_sensor_data)
        self.sync=TimeSynchronizer([self.image,self.info],10);self.sync.registerCallback(self.camera)
        self.timer=self.create_timer(.05,self.tick)
        self.emit('OBSERVER_STARTED',selected_id=self.selected,current_sensor_available=False,commissioning=self.cfg['commissioning'])

    def emit(self,event,**data):
        self.log.write(json.dumps(dict(event=event,monotonic=time.monotonic(),**data),default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x),allow_nan=False)+'\n')

    def joints(self,msg):
        try:
            names=[msg.name.index(f'joint{i}') for i in range(1,5)]
            q=np.asarray([msg.position[i] for i in names]);dq=np.asarray([msg.velocity[i] for i in names])
            grip=msg.name.index('gripper_left_joint');finger=float(msg.position[grip]);finger_v=float(msg.velocity[grip])
            ros_now=self.get_clock().now().nanoseconds/1e9
            stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
            age=ros_now-stamp
            if not 0<=age<.2 or not np.isfinite(np.r_[q,dq,finger,finger_v]).all():return
            self.history.append(RobotSample(time.monotonic()-age,q,dq,finger,finger_v,math.nan,math.nan,(0,0),0))
        except (ValueError,IndexError):return

    def on_goal(self,msg):
        if len(msg.data)==4 and np.isfinite(msg.data).all():self.goal=np.asarray(msg.data);self.planned=time.monotonic()

    def on_action(self,msg):
        if len(msg.data)==4 and np.isfinite(msg.data).all():self.previous=np.asarray(msg.data)

    def status(self,msg):
        self.phase=status_phase(msg.data);self.emit('MP_STATUS',status=msg.data,phase=self.phase.name)
        if 'cargo_loaded:' in msg.data:self.done=True

    def camera(self,image,info):
        try:
            bound=paired_camera_info(image,info,self.cfg['vision']['optical_frame'])
            stamp=image.header.stamp.sec+image.header.stamp.nanosec/1e9
            now=time.monotonic();age=self.get_clock().now().nanoseconds/1e9-stamp
            if not 0<=age<=.4:return
            q=synchronized_joints(self.history,now-age,.06)
            rgb=self.bridge.imgmsg_to_cv2(image,desired_encoding='bgr8')
            k=np.asarray(bound.k).reshape(3,3);d=np.asarray(bound.d)
            detections=self.detector.detect(rgb,k,d);self.cache.begin_frame(now-age)
            for marker,t,r,error in detections:
                center,quat=self.planner.object_from_marker(q,t,r)
                self.cache.update(ObjectPose(marker,now-age,center,quat,error),now)
                if marker==self.selected:
                    raw_pose=PoseStamped();raw_pose.header=image.header
                    raw_pose.pose.position.x,raw_pose.pose.position.y,raw_pose.pose.position.z=map(float,t)
                    quat_cam=Rotation.from_matrix(r).as_quat()
                    raw_pose.pose.orientation.x,raw_pose.pose.orientation.y,raw_pose.pose.orientation.z,raw_pose.pose.orientation.w=map(float,quat_cam)
                    self.aruco_pose.publish(raw_pose)
                    base=np.linalg.inv(self.planner.policy_from_base)@np.r_[center,1.]
                    if self.target_base is None:
                        # Match the existing ArUco bridge's lock-on-first-target
                        # contract. This is a fixed planning GOAL, not fresh PnP.
                        self.target_base=base[:3].copy()
                        self.emit('TARGET_GOAL_LATCHED',selected_id=self.selected,base_goal=self.target_base,source_stamp=now-age)
                    s=.02;corners=np.asarray([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]])
                    uv=cv2.projectPoints(corners,cv2.Rodrigues(r)[0],t,k,d)[0].reshape(4,2)
                    box=Float32MultiArray(data=[float(uv[:,0].min()),float(uv[:,1].min()),float(np.ptp(uv[:,0])),float(np.ptp(uv[:,1]))])
                    self.bbox.publish(box);self.front_bbox.publish(box);self.close_ready.publish(Bool(data=True))
            self.emit('VISION',ids=[x[0] for x in detections],poses=[None if p is None else vars(p) for p in self.cache.items])
            if now-self.saved>=1:
                cv2.imwrite(str(self.args.log_dir/f'frame_{self.count:06d}.png'),rgb);self.saved=now;self.count+=1
        except Exception as exc:self.emit('VISION_REJECTED',reason=str(exc))

    def tick(self):
        now=time.monotonic()
        if not self.history or now-self.history[-1].stamp>.2:return
        sample=self.history[-1];eef=self.planner.forward(sample.q,sample.finger)
        goal=sample.q if self.goal is None else self.goal
        place=np.r_[self.cfg['estimation']['deck_center_xy'],self.cfg['estimation']['deck_top_z']]
        obs=observation(sample,eef,goal,self.phase,0,1,self.cache,self.selected,place,
            False,False,self.previous,self.planned,0,now)
        # A live observation may contain aged/occluded marker slots, exactly as
        # in the model contract. Starting requires only the selected marker;
        # an unobserved other slot retains its invalid/occluded observation.
        self.obs.publish(Float32MultiArray(data=obs.tolist()))
        if self.target_base is not None:
            p=PointStamped();p.header.stamp=self.get_clock().now().to_msg();p.header.frame_id='base_link'
            p.point.x,p.point.y,p.point.z=map(float,self.target_base);self.point.publish(p)
        self.input_ready.publish(Bool(data=self.cache.fresh(self.selected,now,.4)))
        self.emit('OBSERVATION',observation=obs,q=sample.q,dq=sample.dq,finger=sample.finger,phase=self.phase.name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--training',type=Path,required=True);p.add_argument('--scene',type=Path,required=True);p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--selected-id',type=int,choices=(0,1),default=0)
    args,ros_args=p.parse_known_args();rclpy.init(args=ros_args);node=Observer(args)
    try:
        while rclpy.ok() and not node.done:rclpy.spin_once(node,timeout_sec=.1)
    except KeyboardInterrupt:pass
    finally:
        node.emit('EXIT',mission_done=node.done);node.log.close();node.destroy_node()
        if rclpy.ok():rclpy.shutdown()

if __name__=='__main__':main()
