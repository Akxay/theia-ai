#!/usr/bin/env python

import paramiko
import time


def deploy(key, server):
    print("Connecting to box")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, key_filename=key)
    ssh.exec_command('sudo yum install git -y')
    ssh.exec_command('rm -rf theia; '
                     'git clone https://github.com/akkiittiwari/theia.git;')
    print("Launching Flask Server")
    time.sleep(2)
    ssh.exec_command('python $(pwd)/theia/flask_server.py')
    print("Pull from Github successful")
    time.sleep(2)
    print("Script fully executed ... exiting")
    ssh.close()
# deploy('PEM_FILE_PATH', 'ec2-54-212-196-78.us-west-2.compute.amazonaws.com')
