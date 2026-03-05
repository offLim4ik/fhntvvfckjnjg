#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


MAP_FOLDER = "/home/pi/catkin_ws/src/clover/aruco_pose/map"

MAP_FILES = [
    "map1.txt",
    "map2.txt",
    "map3.txt"
]

OUTPUT_MAP = "map.txt"

# ==============================


def read_map(file_path):
    markers = []

    with open(file_path, "r") as f:
        for line in f:

            if line.startswith("#") or line.strip() == "":
                continue

            markers.append(line.strip())

    return markers


def main():

    all_markers = []

    for file in MAP_FILES:

        path = os.path.join(MAP_FOLDER, file)

        if not os.path.exists(path):
            print("Файл не найден:", path)
            continue

        print("Читаю:", file)

        markers = read_map(path)

        all_markers.extend(markers)

    out_path = os.path.join(MAP_FOLDER, OUTPUT_MAP)

    with open(out_path, "w") as f:

        f.write("# id length x y z rot_z rot_y rot_x\n")

        for m in all_markers:
            f.write(m + "\n")

    print("\nГотово.")
    print("Карта сохранена:", out_path)
    print("Всего маркеров:", len(all_markers))


if __name__ == "__main__":
    main()
