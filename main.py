from win32gui import GetWindowText, GetForegroundWindow
from PIL import Image
import dxcam
import simpleaudio as sa
import pytesseract
import numpy as np
import os

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

wave_obj = sa.WaveObject.from_wave_file(os.path.dirname(__file__) + '\\sounds\\alert.wav')

def convert_to_grayscale(image_np, threshold=225):
    img_grey_np = np.array(Image.fromarray(image_np).convert('L'))
    img_grey_np[img_grey_np < threshold] = 0
 
    img_grey_np = 255 - img_grey_np
    # dialate the image
    #img_grey_np = cv2.dilate(img_grey_np, np.ones((3,3), np.uint8), iterations=1)
    #img_grey_np = cv2.erode(img_grey_np, np.ones((3,3), np.uint8), iterations=1)
    try:
      text = pytesseract.image_to_string(img_grey_np, timeout=0.25) # Timeout after half a second , config="--psm 7"
      text = text.strip()
      if (text != ""):
        for fish in sot_fish:
          # use a regex to find the fish name in the text
          if text.lower().find(fish.lower()) != -1:
            print(text)
            # play a sound
            play_obj = wave_obj.play()
            play_obj.wait_done()
    except RuntimeError as timeout_error:
        print('fail')
        # Tesseract processing is terminated
        pass

target_fps = 4
print(dxcam.device_info())
camera = dxcam.create(output_idx=0)
camera.start(target_fps=target_fps, video_mode=False)
while True:
    try:
      frame = camera.get_latest_frame()
      # if (GetWindowText(GetForegroundWindow()) != SOT_WINDOW_NAME):
      #   continue;
      convert_to_grayscale(frame)
      # with Image.fromarray(frame) as img:
        
      #   img.thumbnail((1280, 1280), Image.LANCZOS)
      #   img.save("test.jpg")
    except KeyboardInterrupt:
       break
    except:
       continue
camera.stop()
