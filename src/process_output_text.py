def average_and_output(file_path, row_num=5):
    with open(file_path, "a+") as f:
        # go to the beginning of the file
        f.seek(0)
        # read row_num lines each time
        output_dct = {}
        dist_dct = {}
        while True:
            accs = []
            for i in range(row_num):
                line = f.readline()
                if not line:
                    break
                else:
                    line = line.strip()
                    window_index = line.find("Window_size: ") + len("Window_size: ")
                    window_size = int(line[window_index : window_index + 3].strip(";"))
                    if window_size not in [60, 120]:
                        continue
                    start_index = line.find("Accuracy: ") + len("Accuracy: ")

                    acc = float(line[start_index : start_index + 6])
                    accs.append(acc)
                    dist_index = line.find("Distribution: ") + len("Distribution: ")
                    dist_index_end = line.find("Epoch: ")
                    dist = (
                        line[dist_index:dist_index_end]
                        .strip()
                        .strip("[")
                        .strip("];")
                        .split(" ")
                    )
                    dist = [int(x) for x in dist if x != ""]

                    majority_prop = max(dist) / sum(dist)

                    setting = line[: line.find("Seed") - 1]
                    dist_dct[setting] = round(majority_prop, 4)

            if not line:
                break
            if accs == []:
                continue
            avg_acc = sum(accs) / len(accs)
            output_dct[setting] = round(avg_acc, 4)
    # write to the new file
    output_name = file_path[:-4] + "_avg.txt"
    if "TargetSmoothed" in file_path:
        output_name = file_path[:-4] + "_avg.txt"
    with open(output_name, "w+") as f:
        for setting, avg_acc in output_dct.items():
            f.write(
                setting
                + " Accuracy: "
                + str(avg_acc)
                + " Majority: "
                + str(dist_dct[setting])
                + "\n"
            )


if __name__ == "__main__":
    average_and_output("output_epoch100_lr5e-4_fullval_TargetSmoothed_cleaned.txt")
