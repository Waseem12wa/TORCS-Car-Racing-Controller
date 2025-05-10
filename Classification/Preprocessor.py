import os
import pandas as pd

def collect_txt_files(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for folder in ['dirttracks', 'ovaltracks', 'roadtracks']:
            folder_path = os.path.join(root_dir, folder)
            for dirpath, _, filenames in os.walk(folder_path):
                for file in filenames:
                    if file.endswith('.txt'):
                        file_path = os.path.join(dirpath, file)
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                            outfile.write('\n')  # Optional: separate files with newline
    print(f"All .txt files merged into '{output_file}'.")

def parse(str_sensors):
    sensors = {}
    b_open = str_sensors.find('(')

    while b_open >= 0:
        b_close = str_sensors.find(')', b_open)
        if b_close >= 0:
            substr = str_sensors[b_open + 1: b_close]
            items = substr.split()
            if len(items) < 2:
                print("Problem parsing substring: " + substr)
            else:
                value = []
                for i in range(1, len(items)):
                    value.append(float(items[i]))
                if len(value) == 1:
                    sensors[items[0]] = float(value[0])
                else:
                    for i in range(0, len(value)):
                        sensors[(items[0] + str(i))] = value[i]
            b_open = str_sensors.find('(', b_close)
        else:
            print("Problem parsing sensor string: " + str_sensors)
            return None

    return sensors

def process_dataset(input_file, output_csv):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.split('\t\t')
            if len(tokens) >= 3:
                da = parse(tokens[1])
                cla = parse(tokens[2])
                if da is not None and cla is not None:
                    da['Accelerator'] = cla.get('accel', 0.0)
                    da['Steer'] = cla.get('steer', 0.0)
                    data.append(da)

    df = pd.DataFrame.from_dict(data)
    df.to_csv(output_csv, index=False, header=True)
    print(f"Parsed data saved to '{output_csv}'.")

if __name__ == '__main__':
    root_dir = 'tracks'  # Change to your base directory if needed
    merged_file = 'maindataset.txt'
    output_csv = 'dataset.csv'

    collect_txt_files(root_dir, merged_file)
    process_dataset(merged_file, output_csv)
