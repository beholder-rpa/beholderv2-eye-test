from win32gui import GetWindowText, GetForegroundWindow
from threading import Thread, Lock
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
import argparse
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

keepers = [
    "Umber Splashtail",
    "Bright Pondie",
    "Raven Islehopper",
    "Bone Ancientscale",
    "Bonedust Plentifin",
    "Muddy Wildsplash",
    "Forsaken Devilfish",
    "Jade Battlegill",
    "Sky Battlegill",
    "Rum Battlegill",
    "Sand Battlegill",
    "Bittersweet Battlegill",
    "Snow Wrecker",
    "Blackcloud Wrecker",
    "Moon Wrecker",
    "Ancient Stormfish",
    "Wild Stormfish",
    "Shadow Stormfish",
    "Twilight Stormfish",
    "Old Boot",
    "Old Hat",
    "Old Skull",
    "Fish Bones",
    "Ashen Key",
]

fish_pattern = re.compile(f"^\s*(?P<fish_name>(?P<is_trophy>Trophy)?\s*(" \
                              "(?P<splashtail>(?P<splashtail_prefix>Ruby|Sunny|Indigo|Umber|Seafoam)\s+?Splashtail)|" \
                              "(?P<pondie>(?P<pondie_prefix>Charcoal|Orchid|Bronze|Bright|Moonsky)\s+?Pondie)|" \
                              "(?P<islehopper>(?P<islehopper_prefix>Stone|Moss|Honey|Raven|Amethyst)\s+?Islehopper)|" \
                              "(?P<ancientscale>(?P<ancientscale_prefix>Almond|Sapphire|Smoke|Bone|Starshine)\s+?Ancientscale)|" \
                              "(?P<plentifin>(?P<plentifin_prefix>Olive|Amber|Cloudy|Bonedust|Watery)\s+?Plentifin)|" \
                              "(?P<wildsplash>(?P<wildsplash_prefix>Russet|Sandy|Ocean|Muddy|Coral)\s+Wildsplash)|" \
                              "(?P<devilfish>(?P<devilfish_prefix>Ashen|Seashell|Lava|Forsaken|Firelight)\s+?Devilfish)|" \
                              "(?P<battlegill>(?P<battlegill_prefix>Jade|Sky|Rum|Sand|Bittersweet)\s+?Battlegill)|" \
                              "(?P<wrecker>(?P<wrecker_prefix>Rose|Sun|Blackcloud|Snow|Moon)\s+Wrecker)|" \
                              "(?P<stormfish>(?P<stormfish_prefix>Ancient|Shores|Wild|Shadow|Twilight)\s+Stormfish)|" \
                              "(?P<plunder>(Fish Bones|Old Hat|Old Boot|Old Skull|Ashen Key))" \
                              "))\s*$", re.IGNORECASE)


alert = sa.WaveObject.from_wave_file(os.path.dirname(__file__) + '\\sounds\\fanfare.wav')
error = sa.WaveObject.from_wave_file(os.path.dirname(__file__) + '\\sounds\\error.wav')
keyboard = KeyboardController()
mouse = MouseController()

action_lock = Lock()
fish_found_lock = Lock()
saved_image_lock = Lock()
stats_lock = Lock()

last_found_fish: str = None
last_found_time: datetime = None

last_saved_fish: str = None
last_saved_time: datetime = None

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
    contours, _ = cv2.findContours(dialate, 1, 2)
    for contour in contours:
      x,y,w,h = cv2.boundingRect(contour)
      
      # if the area is greater than 8% of the image, skip it
      if (w*h > image_np.shape[0]*image_np.shape[1]*0.08):
        continue

      # if the aspect ratio is less than 1:2, skip it
      if (w/h < 2):
        continue

      if h > 0.025*image_np.shape[0] and w > 0.04*image_np.shape[1]:
        textAreas.append((x,y,w,h))
 
    if (len(textAreas) == 0):
      return None
    
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
          # use the fish pattern to find the fish name
          match = fish_pattern.match(text)
          if match:
            fish_name = match.group('fish_name')
            #cv2.imwrite("./test-contour.jpg", crop)
            return fish_name
            
      except RuntimeError as timeout_error:
          print("fish_finder: caught exception RuntimeError: " + str(timeout_error) )
          # Tesseract processing is terminated
          pass
    return None

def process_frame(frame, action_lock, fish_found_lock, saved_image_lock, stats_lock, save_image):
  global last_found_fish
  global last_found_time
  global last_saved_fish
  global last_saved_time

  # if (GetWindowText(GetForegroundWindow()) != SOT_WINDOW_NAME):
  #   continue;
  function_start_time = datetime.now()
  fish_name = fish_finder(frame)
  if (fish_name):
    current_time = datetime.now()
    if (last_found_time == None or (current_time - last_found_time).total_seconds() > 8):
      print(fish_name)


      # double-check lock pattern
      if (last_found_time == None or last_found_time < current_time):
        fish_found_lock.acquire()
        if (last_found_time == None or last_found_time < current_time):
          last_found_time = current_time
          last_found_fish = fish_name
        fish_found_lock.release()

      if (save_image and fish_name != last_saved_fish and (last_saved_time == None or (current_time - last_saved_time).total_seconds() > 30)):
        with Image.fromarray(frame) as img:
          filename = f"./fish_images/{fish_name}_{current_time.strftime('%Y%m%d-%H%M%S')}.jpg"
          img.save(filename)
      
      if (last_saved_time == None or last_saved_time < current_time):
        saved_image_lock.acquire()
        if (last_saved_time == None or last_saved_time < current_time):
          last_saved_fish = fish_name
          last_saved_time = current_time
        saved_image_lock.release()

      # If the fish is not a tier 5 fish, recast
      s_tier_fish = False
      for keeper in keepers:
        if (re.search(f"\s*{keeper}\s*", fish_name, re.IGNORECASE)):
          s_tier_fish = True
          break
      
      if (s_tier_fish):
        alert.play()
      else:
        error.play()
        if (action_lock.acquire(blocking=False)):
          recast()
          action_lock.release()

      # update stats
      stats_lock.acquire()
      # append to the stats file
      with open("stats.csv", "a") as stats_file:
        stats_file.write(f"{function_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},{current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},{fish_name}\n")
      stats_lock.release()

  # with Image.fromarray(frame) as img:
  #   img.thumbnail((1280, 1280), Image.LANCZOS)
  #   img.save("test.jpg")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-s", "--save-images",
                      help="save images of fish caught",
                      action="store_true")
  parser.add_argument("-f", "--target-fps", type=int,
                      help="target fps", default=4)
  args = parser.parse_args()

  print(dxcam.device_info())
  camera = dxcam.create(output_idx=0)
  camera.start(target_fps=args.target_fps, video_mode=False)

  while True:
      try:
        frame = camera.get_latest_frame()
        #process_frame(frame, action_lock, fish_lock)
        th = Thread(target = process_frame, args = (frame, action_lock, fish_found_lock, saved_image_lock, stats_lock, args.save_images))
        th.start()
      except KeyboardInterrupt:
        break
      except Exception as e:
        print(e)
        continue
  camera.stop()

if __name__ == "__main__":
  main()