from win32gui import GetWindowText, GetForegroundWindow
from PIL import Image
import dxcam
import simpleaudio as sa
import pytesseract
import numpy as np
import os
import cv2
from time import sleep
from datetime import datetime
import re
import random
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

SOT_WINDOW_NAME = "Sea of Thieves"

sot_fish = [
    "Splashtail",
    "Plentifin",
    "Ancientscale",
    "Wildsplash",
    "Pondie",
    "Islehopper",
    "Devilfish",
    "Battlegill",
    "Wrecker",
    "Stormfish",
]

s_tier_fish_prefixes = [
    "Umber",
    "Bright",
    "Raven",
    "Bone",
    "Bonedust",
    "Muddy",
    "Foresaken",
    "Sand",
    "Snow",
    "Blackcloud",
    "Shadow",
]


alert = sa.WaveObject.from_wave_file(os.path.dirname(__file__) + '\\sounds\\fanfare.wav')
error = sa.WaveObject.from_wave_file(os.path.dirname(__file__) + '\\sounds\\error.wav')
keyboard = KeyboardController()
mouse = MouseController()

def recast():
  mouse.click(Button.right, 1)
  
  # Press left mouse button to cast
  sleep(1.25)
  mouse.press(Button.left)
  delay = random.uniform(0.1, 1.1)
  sleep(delay)
  mouse.release(Button.left)

def fish_finder(image_np, threshold=165):
    img_grey = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(img_grey,threshold,255,0)

    # dialate the image to find clusters of pixels
    kernel = np.ones((15,15),np.uint8)
    dialate = cv2.dilate(thresh,kernel,iterations = 1)

    
    textAreas = []
    contours,hierarchy = cv2.findContours(dialate, 1, 2)
    for cnt in contours:
      x,y,w,h = cv2.boundingRect(cnt)
      
      # if the area is greater than 8% of the image, skip it
      if (w*h > image_np.shape[0]*image_np.shape[1]*0.08):
        continue

      if h > 40 and w > 150 and w > 3*h:
        textAreas.append((x,y,w,h))
 
    invert = cv2.bitwise_not(thresh)

    # sort the text areas by area
    textAreas.sort(key=lambda tup: tup[2]*tup[3], reverse=True)

    for (x,y,w,h) in textAreas:
      # get the text from the image
      crop = invert[y:y+h, x:x+w]
      
      try:
        text = pytesseract.image_to_string(Image.fromarray(crop), timeout=0.20, config='--psm 7') # Timeout after half a second , config="--psm 7"
        text = text.strip()
        if (text != ""):
          for fish in sot_fish:
            # use a regex to find the fish name in the text
            match = re.search(f"(?P<fish_name>(Trophy\s)?\w+\s{fish})", text, re.IGNORECASE)
            if match:
              fish_name = match.group('fish_name')
              cv2.imwrite("./test-contour.jpg", crop)
              return fish_name
      except RuntimeError as timeout_error:
          print("fish_finder: caught exception RuntimeError: " + str(timeout_error) )
          # Tesseract processing is terminated
          pass
    return None

def main():
  target_fps = 4
  print(dxcam.device_info())
  camera = dxcam.create(output_idx=0)
  camera.start(target_fps=target_fps, video_mode=False)

  last_found_fish: str = None
  last_found_time: datetime = None

  last_saved_fish: str = None
  last_saved_time: datetime = None

  while True:
      try:
        frame = camera.get_latest_frame()
        # if (GetWindowText(GetForegroundWindow()) != SOT_WINDOW_NAME):
        #   continue;
        fish_name = fish_finder(frame)
        if (fish_name):
          current_time = datetime.now()
          if (last_found_time == None or (current_time - last_found_time).total_seconds() > 10):
            print(fish_name)
            last_found_fish = fish_name
            last_found_time = current_time
            # If the fish is not a tier 5 fish, recast
            s_tier_fish = False
            for prefix in s_tier_fish_prefixes:
              if (re.search(f"\s*{prefix}\s+", fish_name, re.IGNORECASE)):
                s_tier_fish = True
                break
            
            if (s_tier_fish):
              alert.play()
            else:
              error.play()
              recast()

          if (fish_name != last_saved_fish and (last_saved_time == None or (current_time - last_saved_time).total_seconds() > 30)):
            last_saved_fish = fish_name
            last_saved_time = current_time
            with Image.fromarray(frame) as img:
              filename = f"./fish_images/{fish_name}_{current_time.strftime('%Y%m%d-%H%M%S')}.jpg"
              img.save(filename)
            
        # with Image.fromarray(frame) as img:
        #   img.thumbnail((1280, 1280), Image.LANCZOS)
        #   img.save("test.jpg")
      except KeyboardInterrupt:
        break
      except Exception as e:
        print(e)
        continue
  camera.stop()

if __name__ == "__main__":
  main()