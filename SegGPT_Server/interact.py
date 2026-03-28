"""
@Brief: Simulate a terminal connection to Linux via SSH
"""
import paramiko
import time

hostname = '***.***.***.***'
port = 22
username = '***'
password = '***'
timeout = 10


def runCommand(chanT, command, endSymbol):
    """
    :brief: Interactive command execution
    :param chanT: Interactive shell
    :param command: command string
    :param endSymbol: Customized end symbol used to determine whether an instruction has been executed successfully
    :return: Command execution results
    """
    chanT.send(command + '\n')  
    results = ''
    while True:
        result = chanT.recv(1024).decode('utf-8')
        results += result
        if results[-2:] in endSymbol:  
            break
    re = results.split('\n')[1:]  
    print('\n'.join(re), end='') 
    return re[:-1] 


if __name__ == "__main__":
    # Example
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port, username, password)
    chan = ssh.invoke_shell() 
    chan.settimeout(1000)
    time.sleep(3)  
    loginInfo = chan.recv(1024).decode('utf-8')
    print(loginInfo, end='')
    endSymbol = ['$ ', '> ', '* '] 
    while True:
        command = input()  
        if command == 'quitshell': 
            print('Bye Bye!')
            break
        result = runCommand(chan, command, endSymbol) 







