
import paramiko
import time
import os

hostname = '***.***.***.***'
port = 22
username = '***'
password = '***'
timeout = 10


def sendFile(local_path='../***.avi', remote_path='***.avi'):
    """
    :brief: Send the original video from this device to the server for processing
    :param local_path: The path of the original file on this machine, including the file name, relative path
    :param remote_path: The path of the file saved on the server must include the file name and absolute path
    :return: None
    """
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(local_path, remote_path, confirm=False)
    sftp.close()
    transport.close()


def receiveFile(local_object_path="***.mp4", local_background_path="***.mp4", object_path='***.mp4', background_path='***.mp4'):
    """
    :brief: Receive two videos processed by the server from the server
    :param local_object_path: The path of the target video to be saved locally, with the file name added. Relative path
    :param local_background_path: The path of the background video to be saved locally, with the file name added. Relative path
    :param object_path: The target video path on the server must include the file name    absolute path
    :param background_path: The path to the background video on the server must include the file name    absolute path
    :return: None
    """
    transport = paramiko.Transport((hostname, port))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.get(object_path, local_object_path)
    sftp.get(background_path, local_background_path)
    sftp.close()
    transport.close()


def runCommand(chanT, command, endSymbol):
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
    local_file_name = "***.avi"  
    local_file_path = "./"+local_file_name  
    remote_src_file_path = "***"+local_file_name  

    remote_file_name = "***.avi"  
    remote_object_file_path = "***"+remote_file_name 
    remote_background_file_path = "***"+remote_file_name  
    local_object_file_path = "./"+"***"+remote_file_name 
    local_background_file_path = "./"+"***"+remote_file_name  

    sendFile(local_file_path, remote_src_file_path) 

    ssh = paramiko.SSHClient()  
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
    ssh.connect(hostname, port, username, password)  
    chan = ssh.invoke_shell()  
    chan.settimeout(1000)
    time.sleep(3)  
    loginInfo = chan.recv(1024).decode('utf-8') 
    print(loginInfo, end='')
    endSymbol = ['$ ', '> ', '* '] 

    runCommand(chan, "conda activate ***", endSymbol) 
    runCommand(chan, "cd ***", endSymbol) 
    runCommand(chan, "python seggpt_inference.py --input_video input/"+local_file_name
                     +" --prompt_image refs/src_2.jpg --prompt_target refs/mask_2.jpg"  
                     " --output_dir ./output/", endSymbol)  

    receiveFile(local_object_file_path, local_background_file_path, remote_object_file_path, remote_background_file_path)  

