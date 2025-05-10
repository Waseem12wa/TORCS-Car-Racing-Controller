

class MsgParser(object):
   
    def __init__(self):
        '''Constructor'''

    def parse(self, str_sensors):
        '''Return a dictionary with tags and values from the UDP message'''
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
                        value.append(items[i])
                    sensors[items[0]] = value
                b_open = str_sensors.find('(', b_close)
            else:
                print("Problem parsing sensor string: " + str_sensors)
                return None

        return sensors

    def parse2(self,str_sensors):
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
    
    def stringify(self, dictionary):
        '''Build an UDP message from a dictionary'''
        msg = ''
        
        for key, value in dictionary.items():
            if value != None and value[0] != None:
                msg += '(' + key
                for val in value:
                    msg += ' ' + str(val)
                msg += ')'
        
        return msg