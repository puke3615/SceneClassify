from PIL import Image
import numpy as np
import config


def get_files(dir):
    import os
    if not os.path.exists(dir):
        return []
    if os.path.isfile(dir):
        return [dir]
    result = []
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        result += get_files(sub_path)
    return result


r = 0  # r mean
g = 0  # g mean
b = 0  # b mean

r_2 = 0  # r^2
g_2 = 0  # g^2
b_2 = 0  # b^2

total = 0

files = get_files(config.PATH_TRAIN_IMAGES)
count = len(files)

for i, image_file in enumerate(files):
    print('Process: %d/%d' % (i, count))
    img = Image.open(image_file)
    img = np.asarray(img)
    # img = img.astype('float32') / 255.
    img = img.astype('float32')
    total += img.shape[0] * img.shape[1]

    r += img[:, :, 0].sum()
    g += img[:, :, 1].sum()
    b += img[:, :, 2].sum()

    r_2 += (img[:, :, 0] ** 2).sum()
    g_2 += (img[:, :, 1] ** 2).sum()
    b_2 += (img[:, :, 2] ** 2).sum()

r_mean = r / total
g_mean = g / total
b_mean = b / total

r_var = r_2 / total - r_mean ** 2
g_var = g_2 / total - g_mean ** 2
b_var = b_2 / total - b_mean ** 2

print('Mean is %s' % ([r_mean, g_mean, b_mean]))
print('Var is %s' % ([r_var, g_var, b_var]))
# Mean is [0.4960301824223457, 0.47806493084428053, 0.44767167301470545]
# Var is [0.084966025569294362, 0.082005493489533315, 0.088877477602068156]
