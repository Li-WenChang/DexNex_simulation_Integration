import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber_node')
        
        # Create a subscriber for the RGB image topic
        self.color_subscriber = self.create_subscription(
            Image,
            '/camera/rgb_image',  # The topic name where the RGB image is published
            self.color_image_callback,
            10
        )

        # Create a subscriber for the depth image topic
        self.depth_subscriber = self.create_subscription(
            Image,
            '/camera/depth_image',  # The topic name where the depth image is published
            self.depth_image_callback,
            10
        )

        # Set up Matplotlib for real-time image updates
        #plt.ion()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(15, 8))
        
        # Initial empty images for RGB and Depth
        self.color = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth = np.zeros((480, 640), dtype=np.float32)

        # Set up the imshow objects for displaying the images
        self.color_img = self.ax[0].imshow(self.color)
        self.depth_img = self.ax[1].imshow(self.depth, cmap='gray')
        
        
        self.ax[0].set_title("RGB Image, from Subscriber")
        self.ax[1].set_title("Depth Image, from Subscriber")

    def color_image_callback(self, msg):
        """Callback function to process incoming RGB image and update the plot."""
        
        # Convert the ROS Image message to a NumPy array (RGB)
        self.color = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        
        # Update the RGB image plot
        self.color_img.set_data(self.color)
        
        # Redraw the figure to show the updated image
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.pause(0.01)

    def depth_image_callback(self, msg):
        """Callback function to process incoming depth image and update the plot."""
        
        # Convert the ROS Image message to a NumPy array (Depth)
        self.depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        
        # Update the depth image plot
        self.depth_img.set_data(self.depth)

        

        # Redraw the figure to show the updated depth image
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        plt.pause(0.01)

def main():
    # Initialize ROS 2
    rclpy.init()

    # Create and spin the node
    image_subscriber = ImageSubscriber()

    # Keep the node running and listening to the image topics
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass

    # Clean up and shut down
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
