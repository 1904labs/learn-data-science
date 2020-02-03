import socket

import requests

url = "slack incoming webhook url"

#ip = socket.gethostbyname(socket.getfqdn())

def get_ip_address():
 ip_address = '';
 s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 s.connect(("8.8.8.8",80))
 ip_address = s.getsockname()[0]
 s.close()
 return ip_address

ip = get_ip_address()

requests.post(url=url, json={"text": f"ip: {ip}"})