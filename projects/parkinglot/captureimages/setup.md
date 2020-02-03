I added these two lines to `/etc/rc.local`

```
sudo /usr/bin/python3 /home/pi/post_ip.py &
sudo /usr/bin/python3 /home/pi/capture.py &
```