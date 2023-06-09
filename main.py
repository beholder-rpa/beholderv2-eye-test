from win32gui import GetWindowText, GetForegroundWindow
from threading import Thread, Lock
from PIL import Image
import dxcam
import simpleaudio as sa
import pytesseract
import numpy as np
import math
import os
import cv2
from time import sleep
from datetime import datetime
import re
import random
import argparse
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
from colorama import Style, Fore, init as colorama_init
import concurrent.futures


colorama_init()
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
    #"Umber Splashtail",
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
    # "Moon Wrecker",
    "Ancient Stormfish",
    "Shores Stormfish",
    "Wild Stormfish",
    "Shadow Stormfish",
    "Twilight Stormfish",
    "Old Boot",
    "Old Hat",
    "Old Skull",
    "Fish Bones",
    "Ashen Key",
]

fish_pattern = re.compile(
    f"^\s*(?P<fish_name>(?P<is_trophy>Trophy)?\s*("
    "(?P<splashtail>(?P<splashtail_prefix>Ruby|Sunny|Indigo|Umber|Seafoam)\s+?Splashtail)|"
    "(?P<pondie>(?P<pondie_prefix>Charcoal|Orchid|Bronze|Bright|Moonsky)\s+?Pondie)|"
    "(?P<islehopper>(?P<islehopper_prefix>Stone|Moss|Honey|Raven|Amethyst)\s+?Islehopper)|"
    "(?P<ancientscale>(?P<ancientscale_prefix>Almond|Sapphire|Smoke|Bone|Starshine)\s+?Ancientscale)|"
    "(?P<plentifin>(?P<plentifin_prefix>Olive|Amber|Cloudy|Bonedust|Watery)\s+?Plentifin)|"
    "(?P<wildsplash>(?P<wildsplash_prefix>Russet|Sandy|Ocean|Muddy|Coral)\s+Wildsplash)|"
    "(?P<devilfish>(?P<devilfish_prefix>Ashen|Seashell|Lava|Forsaken|Firelight)\s+?Devilfish)|"
    "(?P<battlegill>(?P<battlegill_prefix>Jade|Sky|Rum|Sand|Bittersweet)\s+?Battlegill)|"
    "(?P<wrecker>(?P<wrecker_prefix>Rose|Sun|Blackcloud|Snow|Moon)\s+Wrecker)|"
    "(?P<stormfish>(?P<stormfish_prefix>Ancient|Shores|Wild|Shadow|Twilight)\s+Stormfish)|"
    "(?P<plunder>(Fish Bones|Old Hat|Old Boot|Old Skull|Ashen Key))"
    "))\s*$",
    re.IGNORECASE,
)


alert = sa.WaveObject.from_wave_file(
    os.path.dirname(__file__) + "\\sounds\\fanfare.wav"
)
error = sa.WaveObject.from_wave_file(os.path.dirname(__file__) + "\\sounds\\error.wav")
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

debug = False


def recast():
    mouse.click(Button.right, 1)

    # Press left mouse button to cast
    sleep(1.25)
    mouse.press(Button.left)
    delay = random.uniform(0.1, 1.1)
    sleep(delay)
    mouse.release(Button.left)

def process_contour(image_np, contour, timeout=0.20):

    x, y, w, h = cv2.boundingRect(contour)
    crop = image_np[y : y + h, x : x + w]
    crop = cv2.bitwise_not(crop)

    region = Image.fromarray(crop)

    try:
        text = pytesseract.image_to_string(region, timeout=timeout, config="--psm 7")  # type: ignore
        text = text.strip()
        found_text = (text, (x, y))
        if text != "":
            # use the fish pattern to find the fish name
            match = fish_pattern.match(text)
            if match:
                fish_name = match.group("fish_name")
                return (contour, fish_name, found_text)
        return (contour, None, found_text)

    except RuntimeError as timeout_error:
        print("find_fish: caught exception RuntimeError: " + str(timeout_error))
        # Tesseract processing is terminated
        pass


def fish_finder(image_np, threshold=225, timeout=0.20):
    # first, resize the image to a width of 1000px
    # scale_percent = 1000 / image_np.shape[1]
    # width = int(image_np.shape[1] * scale_percent)
    # height = int(image_np.shape[0] * scale_percent)
    # dim = (width, height)
    # image_np = cv2.resize(image_np, dim, interpolation=cv2.INTER_AREA)

    found_fish_name = None
    found_fish_contour = None
    #_, thresh = cv2.threshold(image_np, threshold, 255, 0)
    # perform a color mask instead of a threshold
    thresh = cv2.inRange(image_np, np.array([240, 240, 240], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))

    # Save the threshold image for debugging
    if debug:
        cv2.imwrite("./debug-threshold.jpg", thresh)

    # dialate the image to find clusters of pixels
    kernel_size = int(min(image_np.shape[:2]) * 0.025)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dialate = cv2.dilate(thresh, kernel, iterations=1)

    potential_contours = []
    too_small_contours = []
    too_big_contours = []
    invalid_aspect_ratio_contours = []
    too_sparse_contours = []
    not_tall_enough_contours = []
    found_text = []

    all_contours, _ = cv2.findContours(dialate, 1, 2)

    # sort the contours by distance from the center of the image
    all_contours = sorted(
        all_contours,
        key=lambda c: cv2.pointPolygonTest(c, (image_np.shape[1] / 2, image_np.shape[0] / 2), True),
        reverse=True,
    )
    
    if debug:
        cv2.drawContours(image_np, all_contours, -1, (0, 255, 0), 2)

    for contour in all_contours:
        x, y, w, h = cv2.boundingRect(contour)

        # if the area is greater than 3% of the image, skip it
        if w * h > image_np.shape[0] * image_np.shape[1] * 0.03:
            too_big_contours.append(contour)
            continue

        # if the area is less than 0.75% of the image, skip it
        if w * h < image_np.shape[0] * image_np.shape[1] * 0.0020:
            too_small_contours.append(contour)
            continue

        # if the aspect ratio is less than 1:2, skip it
        if w / h < 2:
            invalid_aspect_ratio_contours.append(contour)
            continue

        potential_contours.append(contour)

    if len(potential_contours) > 0:

         with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for contour in potential_contours[:3]:
                future = executor.submit(process_contour, image_np, contour, timeout=timeout)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                contour, fish_name, text = future.result()
                found_text.append(text)
                if fish_name is not None:
                    found_fish_name = fish_name
                    found_fish_contour = contour
                pass

    if debug:
        # Output the percentage of the the width and height of the image relative to the whole image
        if (found_fish_contour is not None) and (found_fish_name is not None):    
            x, y, w, h = cv2.boundingRect(found_fish_contour)
            print(
                f"\twidth: {w}, height: {h}({w/image_np.shape[1]}, {h/image_np.shape[0]})"
            )
            # determine the ratio of non-zero pixels in the filled region
            r = float(cv2.countNonZero(thresh[y : y + h, x : x + w])) / (w * h)
            print(f"\t{r}")

        cv2.drawContours(image_np, potential_contours, -1, (255, 255, 0), 2)
        cv2.drawContours(image_np, too_small_contours, -1, (0, 255, 255), 2)
        cv2.drawContours(image_np, too_big_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(image_np, invalid_aspect_ratio_contours, -1, (255, 0, 255), 2)
        cv2.drawContours(image_np, too_sparse_contours, -1, (220, 0, 220), 2)
        cv2.drawContours(image_np, not_tall_enough_contours, -1, (0, 100, 100), 2)
        for text, (x, y) in found_text:
            cv2.putText(
                image_np,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            # draw the text with the text having a 2 px black border
            cv2.putText(
                image_np,
                text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite("./debug.jpg", image_np)

    return found_fish_name


def process_frame(
    frame,
    action_lock,
    fish_found_lock,
    saved_image_lock,
    stats_lock,
    save_image,
    target_fps=4,
    play_error_sound=True,
):
    global last_found_fish
    global last_found_time
    global last_saved_fish
    global last_saved_time

    # if (GetWindowText(GetForegroundWindow()) != SOT_WINDOW_NAME):
    #   continue;
    function_start_time = datetime.now()
    fish_name = fish_finder(frame, timeout=(60 / target_fps) / 60)
    if fish_name:
        current_time = datetime.now()
        if (
            last_found_time == None
            or (current_time - last_found_time).total_seconds() > 8
        ):
            print(f"{Fore.LIGHTCYAN_EX}Found {fish_name}{Style.RESET_ALL}")

            # double-check lock pattern
            if last_found_time == None or last_found_time < current_time:
                fish_found_lock.acquire()
                if last_found_time == None or last_found_time < current_time:
                    last_found_time = current_time
                    last_found_fish = fish_name
                fish_found_lock.release()

            if (
                save_image
                and fish_name != last_saved_fish
                and (
                    last_saved_time == None
                    or (current_time - last_saved_time).total_seconds() > 30
                )
            ):
                with Image.fromarray(frame) as img:
                    filename = f"./fish_images/{fish_name}_{current_time.strftime('%Y%m%d-%H%M%S')}.jpg"
                    img.save(filename)

            if last_saved_time == None or last_saved_time < current_time:
                saved_image_lock.acquire()
                if last_saved_time == None or last_saved_time < current_time:
                    last_saved_fish = fish_name
                    last_saved_time = current_time
                saved_image_lock.release()

            # If the fish is not a tier 5 fish, recast
            s_tier_fish = False
            for keeper in keepers:
                if re.search(f"\s*{keeper}\s*", fish_name, re.IGNORECASE):
                    s_tier_fish = True
                    break

            if s_tier_fish:
                alert.play()
            else:
                if play_error_sound:
                    error.play()
                if action_lock.acquire(blocking=False):
                    recast()
                    action_lock.release()

            # update stats
            stats_lock.acquire()
            # append to the stats file
            with open("stats.csv", "a") as stats_file:
                stats_file.write(
                    f"{function_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},{current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},{fish_name}\n"
                )
            stats_lock.release()

    # with Image.fromarray(frame) as img:
    #   img.thumbnail((1280, 1280), Image.LANCZOS)
    #   img.save("test.jpg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--save-images", help="save images of fish caught", action="store_true"
    )
    parser.add_argument("-f", "--target-fps", type=int, help="target fps", default=6)
    parser.add_argument(
        "-L",
        "--screen-left",
        type=float,
        help="percent of the left-side of the screen to exclude",
        default=0.20,
    )
    parser.add_argument(
        "-T",
        "--screen-top",
        type=float,
        help="percent of the top of the screen to exclude",
        default=0.40,
    )
    parser.add_argument(
        "-R",
        "--screen-right",
        type=float,
        help="percent of the right of the screen to exclude",
        default=0.20,
    )
    parser.add_argument(
        "-B",
        "--screen-bottom",
        type=float,
        help="percent of the bottom of the screen to exclude",
        default=0.05,
    )
    parser.add_argument("-n", "--no-error-sound", action="store_true")
    parser.add_argument("-d", "--debug", action="store_true")
    args = parser.parse_args()

    global debug
    debug = args.debug
    print(f"Debug: {debug}")

    print(dxcam.device_info())
    print(dxcam.output_info())

    camera = dxcam.create(output_idx=0, output_color="BGR")

    resolution = camera._output.resolution
    left, top = math.floor(resolution[0] * args.screen_left), math.floor(
        resolution[1] * args.screen_top
    )
    right, bottom = resolution[0] - math.floor(
        resolution[0] * args.screen_right
    ), resolution[1] - math.floor(resolution[1] * args.screen_bottom)
    region = (left, top, right, bottom)

    camera.start(target_fps=args.target_fps, region=region, video_mode=False)

    print(f"{Fore.LIGHTMAGENTA_EX}Starting fishing bot...{Style.RESET_ALL}")
    print(
        f"\t{Fore.LIGHTYELLOW_EX}Target FPS: {args.target_fps} ({60/args.target_fps/60:.2f} seconds per frame){Style.RESET_ALL}"
    )
    print(f"\t{Fore.LIGHTYELLOW_EX}Screen focus region: {region}{Style.RESET_ALL}")
    if args.save_images:
        print(f"\t{Fore.LIGHTGREEN_EX}Saving images of fish caught{Style.RESET_ALL}")
    if args.no_error_sound:
        print(
            f"\t{Fore.LIGHTGREEN_EX}Suppressing error sound when a fish is rejected{Style.RESET_ALL}"
        )
    if args.debug:
        print(f"\t{Fore.LIGHTGREEN_EX}Debugging mode enabled{Style.RESET_ALL}")
    print(f"Press Ctrl+C to exit...")

    while True:
        try:
            frame = camera.get_latest_frame()
            process_frame(
                frame,
                action_lock,
                fish_found_lock,
                saved_image_lock,
                stats_lock,
                args.save_images,
                target_fps=args.target_fps,
                play_error_sound=not args.no_error_sound,
            )
            # th = Thread(
            #     target=process_frame,
            #     args=(
            #         frame,
            #         action_lock,
            #         fish_found_lock,
            #         saved_image_lock,
            #         stats_lock,
            #         args.save_images,
            #         args.target_fps,
            #         not args.no_error_sound,
            #     ),
            # )
            # th.start()
        except KeyboardInterrupt:
            print("Exiting...")
            try:
                camera.stop()
            except:
                pass
            exit()
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    main()
