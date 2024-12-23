import argparse
import sys
import numpy as np
import torch
import aria.sdk as aria
import cv2
from typing import Sequence
from projectaria_tools.core.sensor_data import ImageDataRecord, MotionData, BarometerData
from projectaria_tools.core import data_provider
import time
from collections import deque

class GridTest:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_cells = [1, 2, 3, 4, 5, 6]
        self.rows = 2
        self.cols = 3
        self.cell_width = screen_width // self.cols
        self.cell_height = screen_height // self.rows
        self.dwell_time = 1.0  # seconds
        self.current_dwell_start = None
        self.current_cell = None
        self.dwell_progress = 0.0
        
    def get_cell_from_coords(self, x, y):
        col = int(x * self.cols)
        row = int(y * self.rows)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row * self.cols + col + 1
        return None
        
    def draw_grid(self, screen, gaze_point):
        # Draw grid lines
        for i in range(self.cols + 1):
            x = i * self.cell_width
            cv2.line(screen, (x, 0), (x, self.screen_height), (100, 100, 100), 2)
        for i in range(self.rows + 1):
            y = i * self.cell_height
            cv2.line(screen, (0, y), (self.screen_width, y), (100, 100, 100), 2)
            
        # Draw numbers in cells
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(self.rows):
            for j in range(self.cols):
                number = i * self.cols + j + 1
                text_size = cv2.getTextSize(str(number), font, 3, 3)[0]
                x = j * self.cell_width + (self.cell_width - text_size[0]) // 2
                y = i * self.cell_height + (self.cell_height + text_size[1]) // 2
                cv2.putText(screen, str(number), (x, y), font, 3, (255, 255, 255), 3)
                
        # Handle dwell selection
        if gaze_point:
            cell = self.get_cell_from_coords(gaze_point[0], gaze_point[1])
            current_time = time.time()
            
            if cell != self.current_cell:
                self.current_cell = cell
                self.current_dwell_start = current_time
                self.dwell_progress = 0.0
            elif self.current_dwell_start:
                elapsed = current_time - self.current_dwell_start
                self.dwell_progress = min(1.0, elapsed / self.dwell_time)
                
                if elapsed >= self.dwell_time:
                    # Cell selected! Reset progress
                    self.current_dwell_start = None
                    self.dwell_progress = 0.0
                    return True, cell
                
            # Draw progress indicator
            if self.current_cell and self.dwell_progress > 0:
                row = (self.current_cell - 1) // self.cols
                col = (self.current_cell - 1) % self.cols
                x = col * self.cell_width
                y = row * self.cell_height
                
                # Draw progress corners
                progress_length = int(min(self.cell_width, self.cell_height) * 0.3)
                progress_points = [
                    [(x, y), (x + progress_length, y)],
                    [(x, y), (x, y + progress_length)],
                    [(x + self.cell_width, y), (x + self.cell_width - progress_length, y)],
                    [(x + self.cell_width, y), (x + self.cell_width, y + progress_length)],
                    [(x, y + self.cell_height), (x + progress_length, y + self.cell_height)],
                    [(x, y + self.cell_height), (x, y + self.cell_height - progress_length)],
                    [(x + self.cell_width, y + self.cell_height), 
                     (x + self.cell_width - progress_length, y + self.cell_height)],
                    [(x + self.cell_width, y + self.cell_height), 
                     (x + self.cell_width, y + self.cell_height - progress_length)]
                ]
                
                num_segments = int(len(progress_points) * self.dwell_progress)
                for i in range(num_segments):
                    cv2.line(screen, progress_points[i][0], progress_points[i][1], (0, 255, 0), 3)
                    
        return False, None

class BaseStreamingClientObserver:
   def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
       pass

   def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
       pass

   def on_magneto_received(self, sample: MotionData) -> None:
       pass

   def on_baro_received(self, sample: BarometerData) -> None:
       pass

   def on_streaming_client_failure(self, reason, message: str) -> None:
       pass

class EyeTrackingObserver(BaseStreamingClientObserver):
   def __init__(self, inference_model, device_calibration, device="cpu"):
       self.inference_model = inference_model
       self.device_calibration = device_calibration
       self.device = device
       self.value_mapping = {}
       self.calibration_matrix = np.eye(2)
       self.calibration_bias = np.zeros(2)
       self.gaze_history = deque(maxlen=30)
       
   def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
    if record.camera_id == aria.CameraId.EyeTrack:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        image = np.ascontiguousarray(image)
        image_tensor = torch.from_numpy(image).to(self.device)
        
        try:
            preds, lower, upper = self.inference_model.predict(image_tensor)
            preds = preds.detach().cpu().numpy()
            
            # Apply calibration first
            calibrated_preds = np.dot(preds[0], self.calibration_matrix.T) + self.calibration_bias
            
            # Handle X coordinate - left/right (already working well)
            raw_x = float(-preds[0][0])
            cal_x = float(-calibrated_preds[0])
            
            # Handle Y coordinate with corrected scaling and offset
            y_offset = -0.278  # Adjust this offset to center the vertical position
            raw_y = float(-preds[0][1] + y_offset)
            cal_y = float(-calibrated_preds[1] + y_offset)
            
            # Apply vertical scale to match expected range
            vertical_scale = 1.5  # Adjust this to match the desired vertical range
            raw_y *= vertical_scale
            cal_y *= vertical_scale
            
            # Create the final points
            raw_point = (raw_x, raw_y)
            calibrated_point = (cal_x, cal_y)
            
            self.gaze_history.append((raw_point, calibrated_point))
            
            self.value_mapping = {
                "gaze_x": calibrated_point[0],
                "gaze_y": calibrated_point[1],
                "raw_x": raw_point[0],
                "raw_y": raw_point[1]
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")

   def update_calibration(self, target_points, measured_points):
       target_points = np.array(target_points)
       measured_points = np.array(measured_points)
       
       A = np.column_stack((measured_points, np.ones((len(measured_points), 1))))
       b = target_points
       
       solution = np.linalg.lstsq(A, b, rcond=None)[0]
       self.calibration_matrix = solution[:2, :2]
       self.calibration_bias = solution[2, :]

class EyeTrackingVisualizer:
   def __init__(self, screen_width=1920, screen_height=1080):
       self.window_name = "Eye Tracking Visualization"
       self.screen_width = screen_width 
       self.screen_height = screen_height
       self.calibration_points = [
           (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
           (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
           (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
       ]
       self.target_points = []
       self.measured_points = []

   def draw_gaze(self, screen, observer, color=(0, 255, 0)):
    if observer.value_mapping:
        # Scale coordinates to screen space with proper aspect ratio handling
        def scale_coords(x, y):
            screen_x = int((x + 0.5) * self.screen_width)
            screen_y = int((y + 0.5) * self.screen_height)
            return screen_x, screen_y
        
        # Draw raw gaze point
        raw_x, raw_y = scale_coords(
            observer.value_mapping["raw_x"],
            observer.value_mapping["raw_y"]
        )
        
        cv2.circle(screen, (raw_x, raw_y), 10, (0, 0, 255), 2)
        cv2.line(screen, (raw_x - 20, raw_y), (raw_x + 20, raw_y), (0, 0, 255), 2)
        cv2.line(screen, (raw_x, raw_y - 20), (raw_x, raw_y + 20), (0, 0, 255), 2)

        # Draw calibrated gaze point
        cal_x, cal_y = scale_coords(
            observer.value_mapping["gaze_x"],
            observer.value_mapping["gaze_y"]
        )
        
        cv2.circle(screen, (cal_x, cal_y), 10, color, 2)
        cv2.line(screen, (cal_x - 20, cal_y), (cal_x + 20, cal_y), color, 2)
        cv2.line(screen, (cal_x, cal_y - 20), (cal_x, cal_y + 20), color, 2)

        # Draw gaze trail with proper scaling
        points = [scale_coords(x[1][0], x[1][1]) for x in observer.gaze_history]
        if len(points) > 1:
            for i in range(1, len(points)):
                alpha = i / len(points)
                cv2.line(screen, points[i-1], points[i],
                        (int(color[0]*alpha), int(color[1]*alpha), int(color[2]*alpha)), 1)

   def draw_main_menu(self):
       screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       
       # Draw title with glow effect
       title = "Social Eyes Model Test"
       title_size = cv2.getTextSize(title, font, 2.0, 3)[0]
       title_x = (self.screen_width - title_size[0]) // 2
       
       for offset in range(5, 0, -1):
           alpha = offset / 5.0
           color = (int(50*alpha), int(50*alpha), int(100*alpha))
           cv2.putText(screen, title, (title_x, 100), font, 2.0, color, 6-offset)
       cv2.putText(screen, title, (title_x, 100), font, 2.0, (255, 255, 255), 3)
       
       menu_items = [
            ("1: Start Direction Prediction Test", (0, 255, 0)),
            ("2: Start Calibration", (0, 255, 255)),
            ("3: Start Grid Test", (0, 255, 0)),  # Add this line
            ("ESC: Exit", (255, 0, 0))
        ]
       
       y_offset = self.screen_height // 3
       for item, color in menu_items:
           text_size = cv2.getTextSize(item, font, 1.5, 2)[0]
           text_x = (self.screen_width - text_size[0]) // 2
           cv2.putText(screen, item, (text_x, y_offset), font, 1.5, color, 2)
           y_offset += 80
           
       cv2.imshow(self.window_name, screen)

   def run_calibration(self, observer):
       self.target_points = []
       self.measured_points = []
       
       for i, point in enumerate(self.calibration_points):
           screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
           
           # Convert normalized coordinates to screen coordinates
           target_x = int(point[0] * self.screen_width)
           target_y = int(point[1] * self.screen_height)
           
           # Draw instructions
           font = cv2.FONT_HERSHEY_SIMPLEX
           cv2.putText(screen, f"Point {i+1}/9: Focus on the green circle", 
                      (50, 50), font, 1.0, (255, 255, 255), 2)
           cv2.putText(screen, "Press SPACE to confirm or ESC to cancel", 
                      (50, 100), font, 1.0, (255, 255, 255), 2)
           
           # Animation loop
           start_time = time.time()
           while True:
               screen_copy = screen.copy()
               
               # Draw animated target point
               pulse = (np.sin(time.time() * 4) + 1) / 2
               radius = int(20 + pulse * 10)
               cv2.circle(screen_copy, (target_x, target_y), radius, (0, 255, 0), 2)
               cv2.circle(screen_copy, (target_x, target_y), 5, (0, 255, 0), -1)
               
               # Draw current gaze
               self.draw_gaze(screen_copy, observer)
               
               cv2.imshow(self.window_name, screen_copy)
               key = cv2.waitKey(1) & 0xFF
               
               if key == 27:  # ESC
                   return False
               elif key == 32:  # SPACE
                   if observer.value_mapping:
                       self.target_points.append([point[0], point[1]])
                       self.measured_points.append([
                           observer.value_mapping["raw_x"],
                           observer.value_mapping["raw_y"]
                       ])
                   break
           
       # Update calibration matrix
       observer.update_calibration(self.target_points, self.measured_points)
       
       # Show calibration results with test point
       
       # Show calibration results with test point
       screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       
       # Calculate and show accuracy
       error = 0
       for target, measured in zip(self.target_points, self.measured_points):
           calibrated = np.dot(measured, observer.calibration_matrix.T) + observer.calibration_bias
           error += np.sqrt(np.sum((np.array(target) - calibrated) ** 2))
       avg_error = error / len(self.target_points)
       
       # Draw test point in center
       center_x = self.screen_width // 2
       center_y = self.screen_height // 2
       
       start_time = time.time()
       while time.time() - start_time < 5.0:
           screen_copy = screen.copy()
           
           # Animated test point
           pulse = (np.sin(time.time() * 4) + 1) / 2
           radius = int(20 + pulse * 10)
           cv2.circle(screen_copy, (center_x, center_y), radius, (0, 255, 0), 2)
           cv2.circle(screen_copy, (center_x, center_y), 5, (0, 255, 0), -1)
           
           # Draw current gaze
           self.draw_gaze(screen_copy, observer)
           
           # Show calibration stats
           cv2.putText(screen_copy, f"Calibration Complete!", 
                      (50, 50), font, 1.5, (0, 255, 0), 2)
           cv2.putText(screen_copy, f"Average Error: {avg_error:.3f}", 
                      (50, 100), font, 1.0, (255, 255, 255), 2)
           cv2.putText(screen_copy, "Testing calibration... Look at the center point", 
                      (50, 150), font, 1.0, (255, 255, 255), 2)
           
           cv2.imshow(self.window_name, screen_copy)
           if cv2.waitKey(1) & 0xFF == 27:
               break
       
       return True

   def run_grid_test(self, observer):
       grid_test = GridTest(self.screen_width, self.screen_height)
       
       while True:
           screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
           
           # Get current gaze point
           gaze_point = None
           if observer.value_mapping:
               gaze_point = (
                   observer.value_mapping["raw_x"] + 0.5,
                   observer.value_mapping["raw_y"] + 0.5
               )
               
           # Draw grid and handle selection
           selected, cell = grid_test.draw_grid(screen, gaze_point)
           if selected:
               # Flash selected cell
               flash_screen = screen.copy()
               row = (cell - 1) // 3
               col = (cell - 1) % 3
               x = col * (self.screen_width // 3)
               y = row * (self.screen_height // 2)
               cv2.rectangle(flash_screen, (x, y), 
                           (x + self.screen_width // 3, y + self.screen_height // 2),
                           (0, 255, 0), -1)
               cv2.addWeighted(flash_screen, 0.3, screen, 0.7, 0, screen)
           
           # Draw current gaze
           self.draw_gaze(screen, observer)
           
           # Show instructions
           font = cv2.FONT_HERSHEY_SIMPLEX
           cv2.putText(screen, "Look at a number and hold gaze to select", 
                      (50, 30), font, 1.0, (255, 255, 255), 2)
           cv2.putText(screen, "Press ESC to return to menu", 
                      (50, 70), font, 1.0, (255, 255, 255), 2)
           
           cv2.imshow(self.window_name, screen)
           if cv2.waitKey(1) & 0xFF == 27:
               break

   def visualize_predictions(self, observer):
       screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
       font = cv2.FONT_HERSHEY_SIMPLEX
       
       # Draw grid
       for x in range(0, self.screen_width, 100):
           cv2.line(screen, (x, 0), (x, self.screen_height), (20, 20, 20), 1)
       for y in range(0, self.screen_height, 100):
           cv2.line(screen, (0, y), (self.screen_width, y), (20, 20, 20), 1)
           
       # Draw current gaze
       self.draw_gaze(screen, observer)
       
       # Show values
       if observer.value_mapping:
           cv2.putText(screen, f"Raw X: {observer.value_mapping['raw_x']:.3f}", 
                      (50, 50), font, 1.0, (0, 0, 255), 2)
           cv2.putText(screen, f"Raw Y: {observer.value_mapping['raw_y']:.3f}", 
                      (50, 100), font, 1.0, (0, 0, 255), 2)
           cv2.putText(screen, f"Calibrated X: {observer.value_mapping['gaze_x']:.3f}", 
                      (50, 150), font, 1.0, (0, 255, 0), 2)
           cv2.putText(screen, f"Calibrated Y: {observer.value_mapping['gaze_y']:.3f}", 
                      (50, 200), font, 1.0, (0, 255, 0), 2)

       cv2.putText(screen, "Press ESC to return to menu", 
                  (50, self.screen_height - 30), font, 1.0, (255, 255, 255), 2)
       
       cv2.imshow(self.window_name, screen)

   def run(self, observer):
       cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
       cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
       
       current_mode = "main_menu"
       
       while True:
           if current_mode == "main_menu":
               self.draw_main_menu()
               key = cv2.waitKey(0) & 0xFF
               
               if key == ord('1'):
                   current_mode = "visualization"
               elif key == ord('2'):
                   current_mode = "calibration"
               elif key == ord('3'):
                   current_mode = "grid_test"
               elif key == 27:
                   break
                   
           elif current_mode == "visualization":
               self.visualize_predictions(observer)
               if cv2.waitKey(1) & 0xFF == 27:
                   current_mode = "main_menu"
                   
           elif current_mode == "calibration":
               if self.run_calibration(observer):
                   current_mode = "main_menu"
               else:
                   break
                   
           elif current_mode == "grid_test":
               self.run_grid_test(observer)
               current_mode = "main_menu"
                   
       cv2.destroyWindow(self.window_name)

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--interface", dest="streaming_interface", type=str, required=True,
                     choices=["usb", "wifi"])
   parser.add_argument("--model_checkpoint_path", type=str, required=True)
   parser.add_argument("--model_config_path", type=str, required=True)
   parser.add_argument("--device", type=str, default="cpu")
   parser.add_argument("--device-ip", help="IP address for WiFi connection")
   args = parser.parse_args()

   from inference import infer
   inference_model = infer.EyeGazeInference(args.model_checkpoint_path,
                                         args.model_config_path, args.device)

   device_client = aria.DeviceClient()
   client_config = aria.DeviceClientConfig()
   
   if args.device_ip:
       client_config.ip_v4_address = args.device_ip
   device_client.set_client_config(client_config)
   
   device = device_client.connect()
   streaming_manager = device.streaming_manager
   streaming_client = streaming_manager.streaming_client
   
   streaming_config = aria.StreamingConfig()
   if args.streaming_interface == "usb":
       streaming_config.streaming_interface = aria.StreamingInterface.Usb
   streaming_config.security_options.use_ephemeral_certs = True
   
   streaming_manager.streaming_config = streaming_config
   streaming_manager.start_streaming()

   provider = data_provider.create_vrs_data_provider("44859ce1-717c-418f-be4c-ca5aad11e4e3.vrs")
   device_calibration = provider.get_device_calibration()

   observer = EyeTrackingObserver(inference_model, device_calibration, args.device)
   streaming_client.set_streaming_client_observer(observer)
   streaming_client.subscribe()

   visualizer = EyeTrackingVisualizer()
   visualizer.run(observer)

   streaming_client.unsubscribe()
   streaming_manager.stop_streaming()
   device_client.disconnect(device)

if __name__ == "__main__":
   main()