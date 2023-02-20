import threading

import rclpy

rclpy.init()
node = rclpy.create_node('simple_node')

# Spin in a separate thread
thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
thread.start()

rate = node.create_rate(2)

try:
    while rclpy.ok():
        print('Help me body, you are my only hope')
        rate.sleep()
except KeyboardInterrupt:
    pass

rclpy.shutdown()
thread.join()