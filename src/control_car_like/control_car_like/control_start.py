import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from .lib.drive_bot import Navigator
from nav_msgs.msg import Odometry
from numpy import interp
from sensor_msgs.msg import LaserScan

from .tracker.track import *


class maze_solver(Node):

    def __init__(self):

        super().__init__("control_car")

        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.videofeed_subscriber = self.create_subscription(
            Image, '/upper_camera/image_raw', self.get_video_feed_cb, 10)
        self.bot_subscriber = self.create_subscription(
            Image, '/camera_red/image_raw', self.process_data_bot, 10)
        self.timer_period = 0.5
        self.timer = self.create_timer(self.timer_period, self.send_cmd_vel)

        self.bridge = CvBridge()
        self.vel_msg = Twist()

        self.navigator = Navigator()
        self.pose_subscriber = self.create_subscription(
            Odometry, '/odom', self.navigator.bot_motionplanner.get_pose, 10)

        self.sat_view = []
        self.prius_dashcam = []
        self.angle_new = 0.0
        self.speed_new = 0.0
        self.flag = None

    def send_cmd_vel(self):
        self.velocity_publisher.publish(self.vel_msg)

    def control_car(self, box_obj, areas, image):
        for index, area in enumerate(areas):
            if areas[area] > 90000:
                # print()
                angle = interp((box_obj[index][0][0]-320),
                               (-320, 320), (-80, 80))
                if self.angle_new*angle > 0:
                    self.angle_new = angle
                else:
                    self.speed_new = -1.0
                    self.angle_new = 0.0
                # cv2.circle(image, box_obj[index][0], 10, (0,0,255), -1)

    def get_video_feed_cb(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self.sat_view = frame
            self.navigator.navigate_to_home(self.sat_view)
            self.speed_new = float(
                self.navigator.bot_motionplanner.vel_linear_x)
            self.angle_new = self.navigator.bot_motionplanner.vel_angular_z
            cv2.imshow("img_upper_camera", frame)
            cv2.waitKey(1)
        except:
            pass


    def process_data_bot(self, data):

        self.bot_view = self.bridge.imgmsg_to_cv2(
            data, 'bgr8')  # performing conversion
        box_obj, img_ori, areas = tracking_sort(self.bot_view)
        self.control_car(box_obj, areas, img_ori)

        self.vel_msg.angular.z = self.angle_new
        self.vel_msg.linear.x = self.speed_new
        self.velocity_publisher.publish(self.vel_msg)
        cv2.imshow("Bot_view", img_ori)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init()
    node_obj = maze_solver()
    rclpy.spin(node_obj)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
