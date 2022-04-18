import glob
from PIL import Image

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def make_gif(frame_folder):
    frames = [Image.open(image) for image in sorted(glob.glob('Figure_*.png'), key=lambda name: int(name[7:-4]))]
    frame_one = frames[0]
    frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
               save_all=True, duration=200, loop=0)
    
if __name__ == "__main__":
    #print(glob.glob(f"./Figure_*.png"))
    # for file in sorted(glob.glob('Figure_*.png'), key=lambda name: int(name[7:-4])):
        # print(file)
    make_gif(".")