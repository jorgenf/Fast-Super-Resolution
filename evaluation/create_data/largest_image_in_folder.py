from PIL import Image
import glob

INPUT_FRAMES_FOLDER = "evaluation_images/isolated_tags/charuco_CH1_35-15_None"

frame_paths = glob.glob(INPUT_FRAMES_FOLDER + "/*.jp*g")
if len(frame_paths) == 0:
    print(f"No .jpg/.jpeg files in {INPUT_FRAMES_FOLDER}")

sizes = [Image.open(f, 'r').size for f in frame_paths]
print(max(sizes))