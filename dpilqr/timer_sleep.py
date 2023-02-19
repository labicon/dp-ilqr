import threading
from time import perf_counter
import rclpy
import time


def set_sleep_rate(frequency):
    rclpy.init()
    node = rclpy.create_node('simple_node')

    # Spin in a separate thread
    thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    thread.start()

    rate = node.create_rate(frequency)
    return rate

    # try:
    #     while rclpy.ok():
    #         # time.sleep(0.6)
    #         # print('Help me body, you are my only hope')
    #         t0 = perf_counter()
    #         rate.sleep()
    #         tf = perf_counter()
    #         # print(f'{(tf-t0)} seconds passed')
    # except KeyboardInterrupt:
    #     pass

    # rclpy.shutdown()
    # thread.join()
    