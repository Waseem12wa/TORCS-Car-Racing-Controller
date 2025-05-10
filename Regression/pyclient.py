#!/usr/bin/env python
'''
Python client for connecting to the TORCS SCRC server
'''
import sys
import argparse
import socket
import time
import driver

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')

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
parser.add_argument('--verbose', action='store_true', dest='verbose', default=False,
                    help='Enable verbose output')

def main():
    """Main function to start the client"""
    arguments = parser.parse_args()

    # Print connection summary
    print('='*50)
    print('TORCS Neural Network Client')
    print('='*50)
    print(f'Connecting to server: {arguments.host_ip}:{arguments.host_port}')
    print(f'Bot ID: {arguments.id}')
    print(f'Maximum episodes: {arguments.max_episodes}')
    print(f'Maximum steps: {arguments.max_steps}')
    print(f'Track: {arguments.track}')
    print(f'Stage: {arguments.stage}')
    print(f'Verbose mode: {"Enabled" if arguments.verbose else "Disabled"}')
    print('='*50)

    try:
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set timeout for receiving data
        sock.settimeout(1.0)
        print("Socket created successfully")
    except socket.error as msg:
        print(f'Failed to create socket: {msg}')
        sys.exit(-1)

    # Initialize client variables
    shutdownClient = False
    curEpisode = 0
    
    # Create driver instance
    d = driver.Driver(arguments.stage, str(arguments.track))
    print("Driver initialized")

    # Main client loop
    while not shutdownClient:
        # Connection loop
        connection_attempts = 0
        max_connection_attempts = 5
        
        while True:
            if connection_attempts >= max_connection_attempts:
                print(f"Failed to connect after {max_connection_attempts} attempts. Exiting...")
                sys.exit(-1)
                
            print(f'Sending ID to server: {arguments.id}')
            buf = arguments.id + d.init()
            
            if arguments.verbose:
                print(f'Sending init string: {buf}')
            
            try:
                sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print(f"Failed to send data: {msg}")
                sys.exit(-1)
                
            try:
                buf, addr = sock.recvfrom(1000)
                buf = buf.decode('utf-8')
            except socket.timeout:
                print("Connection timed out, retrying...")
                connection_attempts += 1
                continue
            except socket.error as msg:
                print(f"Socket error: {msg}")
                connection_attempts += 1
                continue
        
            if 'identified' in buf:
                print(f'Successfully connected to TORCS server: {buf}')
                break
            else:
                print(f"Unexpected response: {buf}")
                connection_attempts += 1

        # Racing loop
        currentStep = 0
        print("Starting racing loop")

        while True:
            # Wait for data from server
            buf = None
            try:
                data, addr = sock.recvfrom(1000)
                buf = data.decode('utf-8')
            except socket.timeout:
                print("No data received from server...")
                continue
            except socket.error as msg:
                print(f"Socket error: {msg}")
                break

            if arguments.verbose:
                print(f'Received: {buf}')
            
            # Check for shutdown or restart commands
            if buf and 'shutdown' in buf:
                d.onShutDown()
                shutdownClient = True
                print('Client shutdown requested by server')
                break
            
            if buf and 'restart' in buf:
                d.onRestart()
                print('Client restart requested by server')
                break
            
            # Process current step
            currentStep += 1
            
            # Check if we've reached the maximum steps
            if arguments.max_steps > 0 and currentStep >= arguments.max_steps:
                print(f"Reached maximum steps: {arguments.max_steps}")
                buf = '(meta 1)'
            elif buf:
                # Drive the car and get control message
                buf = d.drive(buf)
            
            if arguments.verbose:
                print(f'Sending: {buf}')
            
            # Send control message back to server
            if buf:
                try:
                    sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
                except socket.error as msg:
                    print(f"Failed to send data: {msg}")
                    break
        
        # Increment episode counter
        curEpisode += 1
        
        # Check if we've reached the maximum episodes
        if arguments.max_episodes > 0 and curEpisode >= arguments.max_episodes:
            print(f"Completed {curEpisode} episodes, shutting down")
            shutdownClient = True

    # Close socket connection
    print("Closing socket connection")
    sock.close()
    print("Client terminated")

if __name__ == "__main__":
    main()