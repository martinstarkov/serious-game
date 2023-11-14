# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import time
import random
import math

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

# finger detection
class landmarker_and_result():
   def __init__(self):
      self.result = mp.tasks.vision.HandLandmarkerResult
      self.landmarker = mp.tasks.vision.HandLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      dirname = os.path.dirname(__file__)
      task_filename = os.path.join(dirname, 'hand_landmarker.task')
      # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
      options = mp.tasks.vision.HandLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path=task_filename), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 2, # track both hands
         min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
         min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
         min_tracking_confidence = 0.3, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()

# drawing all finger joints
def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())
         return annotated_image
   except:
      return rgb_image

# drawing finger tips only
def draw_tips_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            index = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
            middle = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring = hand_landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky = hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]
            
            tips = [index, thumb, middle, ring, pinky]

            #print(tips)

            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in tips])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto)

         return annotated_image
   except:
      return rgb_image

class circle():
   def __init__(self):
      self.r = 0
      self.x = 0
      self.y = 0
      self.thickness = 3
      self.lifetime = 3 # seconds
      self.color = (0, 0, 255)
      self.creation_time = time.monotonic()
      self.tag_for_kill = False

   def __eq__(self, other):
      return self.creation_time == other.creation_time and \
             self.x == other.x and \
             self.y == other.y and \
             self.r == other.r

   # seconds
   def time_alive(self):
      return time.monotonic() - self.creation_time
   
   # 0.0 to 1.0
   def life_fraction(self):
      if self.is_alive():
         f = (self.lifetime - self.time_alive()) / self.lifetime
         assert f >= 0.0 and f <= 1.0
         return f
      else:
         return 0.0
   
   def is_alive(self):
      if self.time_alive() > self.lifetime or self.tag_for_kill:
         return False
      return True
   
# circle generation and management
class path_generator():
   def __init__(self, width, height, circle_delay):
      self.circle_delay = circle_delay # seconds
      self.circles = []
      self.width = width
      self.height = height
      self.radius_range = [10, 30] # seconds
      assert self.radius_range[0] < self.radius_range[1]

   def generate_circle(self):
      c = circle()
      c.r = random.randint(self.radius_range[0], self.radius_range[1])
      c.lifetime = random.randint(1, 5) # seconds
      c.x = random.randint(c.r, self.width - c.r)
      c.y = random.randint(c.r, self.height - c.r)
      print("Generating circle with radius ", c.r, " at (", c.x, ",", c.y, ") id: ", len(self.circles))
      self.circles.append(c)
   
   def update_circles(self):
      new_circles = [] 
      for id, circle in enumerate(self.circles):
         if circle.is_alive():
            new_circles.append(circle)
         else:
            print("Deleting circle with id: ", id)
      self.circles = new_circles

   def tag_circle(self, circle):
      circle.tag_for_kill = True

   def draw_circles(self, image):
      for c in self.circles:
         if c.is_alive():
            r = round(c.r * c.life_fraction())
            cv2.circle(image, (c.x, c.y), r, c.color, thickness=c.thickness, lineType=8, shift=0)

# generic circle vs circle overlap test
def circle_collision(c1, c2):
   dist2 = (c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2
   r2 = (c1.r + c2.r) ** 2
   return dist2 <= r2

# https://github.com/google/mediapipe/blob/a38467bae0355fb3737f66da2743f975624e05e5/mediapipe/python/solutions/drawing_utils.py#L49
# convert 0 to 1 pixels to 0 to width/height pixels
def normalized_to_pixel(normalized_x: float, normalized_y: float, 
                        image_width: int, image_height: int):

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

# check for finger tip and circle overlaps and tag circles for destruction
def update_collisions(pg, detection_result: mp.tasks.vision.HandLandmarkerResult):
   try:
      if detection_result.hand_landmarks != []:
         hand_landmarks_list = detection_result.hand_landmarks

         fingers = []
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            index = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = hand_landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
            middle = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring = hand_landmarks[mp.solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky = hand_landmarks[mp.solutions.hands.HandLandmark.PINKY_TIP]
            fingers += [index, thumb, middle, ring, pinky]

         for finger in fingers:
            fc = circle()
            fc.r = 20 # hitbox forgiveness
            fc.x, fc.y = normalized_to_pixel(finger.x, finger.y, pg.width, pg.height)
            #print("Finger: (", fc.x, ",", fc.y, ")")
            for id, c in enumerate(pg.circles):
               if circle_collision(fc, c):
                  print("Finger collided with circle: ", id)
                  c.color = (0, 255, 0)
                  pg.tag_circle(c)
   except:
      pass

def main():
   # access webcam
   cap = cv2.VideoCapture(0)

   if cap.isOpened(): 
      width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
      height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

      # create landmarker
      hand_landmarker = landmarker_and_result()
      
      pg = path_generator(width, height, circle_delay=1)

      starttime = time.time()
      lasttime = starttime
      lapnum = 0

      while True:
         # pull frame
         ret, frame = cap.read()
         # mirror frame
         frame = cv2.flip(frame, 1)

         # display frame
         # update landmarker results
         hand_landmarker.detect_async(frame)

         result = hand_landmarker.result

         laptime = round((time.time() - lasttime), 2)

         # generate a circle every pg.circle_delay
         if laptime > pg.circle_delay:
            pg.generate_circle()
            lasttime = time.time()
            lapnum += 1

         pg.update_circles()

         update_collisions(pg, result)

         pg.draw_circles(frame)

         frame = draw_tips_on_image(frame, result)
         # draw all joints on frame
         #frame = draw_landmarks_on_image(frame, result)
         if cv2.waitKey(1) == ord('q'):
            break
         cv2.imshow('frame',frame)  

      hand_landmarker.close()

   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()