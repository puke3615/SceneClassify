import os


def calculate_file_num(dir):
    if not os.path.exists(dir):
        return 0
    if os.path.isfile(dir):
        return 1
    count = 0
    for subdir in os.listdir(dir):
        sub_path = os.path.join(dir, subdir)
        count += calculate_file_num(sub_path)
    return count


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
