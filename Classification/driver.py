
from time import sleep

import joblib
import msgParser
import carControl
import carState
import numpy as np

steer = 0

class Driver(object):
    '''
    A driver object for the SCRC
    '''

    def __init__(self, stage, track):
        '''Constructor'''
        self.parser = msgParser.MsgParser()
        self.control = carControl.CarControl()
        self.state = carState.CarState()
        self.model = joblib.load('accel.pkl')
        self.model2 = joblib.load('steer.pkl')
        self.forward = False
        self.io = 0

    def init(self):
        '''Return init string with rangefinder angles'''
        self.angles = [0 for x in range(19)]

        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15

        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5

        return self.parser.stringify({'init': self.angles})

    def drive(self, msg):

        self.state.setFromMsg(msg)

        parsemessage = self.parser.parse2(msg)

        for i in range(0, 36):
            parsemessage.pop('opponents' + str(i))

        parsemessage.pop('focus0')
        parsemessage.pop('focus1')
        parsemessage.pop('focus2')
        parsemessage.pop('focus3')
        parsemessage.pop('focus4')
        parsemessage.pop('damage')
        parsemessage.pop('fuel')
        parsemessage.pop('lastLapTime')
        parsemessage.pop('racePos')

        data = list(parsemessage.values())
        array = np.array(data)
        array2 = array.reshape(1, -1)
        self.control.setSteer(0)
        if self.io == 10:
            output = self.model.predict(array2)
            output2 = self.model2.predict(array2)
            self.control.setAccel(output[0])
            global steer
            if output2[0] == 1:
                steer += 0.1
                self.control.setSteer(steer)
            elif output2[0] == -1:
                steer += 0.1
                self.control.setSteer(-steer)
            else:
                steer = 0
                self.control.setSteer(steer)

            self.io = 0

        self.io += 1

        self.gear()
        print(self.control.toMsg())
        return self.control.toMsg()

    def gear(self):

        rpm = self.state.getRpm()
        gear = self.state.getGear()

        if rpm > 7000:
            gear += 1

        if rpm < 4000 and self.forward == False and gear > 1:
            gear -= 1

        self.control.setGear(gear)

    def onShutDown(self):
        pass

    def onRestart(self):
        pass
