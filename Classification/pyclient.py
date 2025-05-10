#!/usr/bin/env python
'''
Created on Apr 4, 2012

@author: lanquarden
'''
import sys
import argparse
import socket
import driver

if __name__ == '_main_':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description = 'Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='127.0.0.1',
                    help='Host IP address (default: 127.0.0.1)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

arguments = parser.parse_args()

# Print summary
print('Connecting to server host ip:'+ str(arguments.host_ip) + '@ port:' + str(arguments.host_port))
print('Bot ID:' + str(arguments.id))
print('Maximum episodes:' + str(arguments.max_episodes))
print('Maximum steps:' + str(arguments.max_steps))
print('Track:' + str(arguments.track))
print('Stage:' + str(arguments.stage))
print('*')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.')
    sys.exit(-1)

# one second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

verbose = False

d = driver.Driver(arguments.stage, str(arguments.track))

while not shutdownClient:
    while True:
        print('Sending id to server: ' + arguments.id)
        buf = arguments.id + d.init()
        print('Sending init string to server:' + buf)
        
        try:
            sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...")
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
        except socket.error as msg:
            print("didn't get response from server...")
    
        if str(buf).find('identified') >= 0:
            print('Received: ' + str(buf))
            break

    currentStep = 0

    while True:
        # wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
        except socket.error as msg:
            print("didn't get response from server...")

        if verbose:
            print('Received: '+ str(buf))
        
        if buf != None and str(buf).find('shutdown') >= 0:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break
        
        if buf != None and str(buf).find('restart') >= 0:
            d.onRestart()
            print('Client Restart')
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf != None:
                buf = d.drive(str(buf))
        else:
            buf = '(meta 1)'
        
        if verbose:
            print('Sending: '+ str(buf))
        
        if buf != None:
            try:
                sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...")
                sys.exit(-1)
    
    curEpisode += 1
    
    if curEpisode == arguments.max_episodes:
        shutdownClient = True
        

sock.close()