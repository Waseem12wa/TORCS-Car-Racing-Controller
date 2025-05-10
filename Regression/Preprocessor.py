import pandas as pd

def parse(str_sensors):
    sensors = {}
    b_open = str_sensors.find('(')

    while b_open >= 0:
        b_close = str_sensors.find(')', b_open)
        if b_close >= 0:
            substr = str_sensors[b_open + 1: b_close]
            items = substr.split()
            if len(items) < 2:
                print("Problem parsing substring:", substr)
            else:
                value = []
                for i in range(1, len(items)):
                    try:
                        value.append(float(items[i]))
                    except ValueError:
                        print("Invalid float in substring:", substr)
                        continue
                if len(value) == 1:
                    sensors[items[0]] = value[0]
                else:
                    for i in range(len(value)):
                        sensors[f"{items[0]}{i}"] = value[i]
            b_open = str_sensors.find('(', b_close)
        else:
            print("Problem parsing sensor string:", str_sensors)
            return None
    return sensors

if __name__ == '__main__':
    data = []
    with open('maindataset.txt', 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split('\t\t')
            if len(tokens) >= 3:
                da = parse(tokens[1])
                cla = parse(tokens[2])
                if da is not None and cla is not None:
                    da['Accelerator'] = cla.get('accel', 0.0)
                    da['Steer'] = cla.get('steer', 0.0)
                    data.append(da)
                else:
                    print("Skipping due to parsing failure.")
            else:
                print("Skipping malformed line:", line.strip())

    df = pd.DataFrame.from_dict(data)
    df.to_csv('dataset.csv', index=False, header=True)
