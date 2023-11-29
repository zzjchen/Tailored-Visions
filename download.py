# Code by zzjchen
# This code includes functions for downloading user history image generation results.
import os
import subprocess
from subprocess import PIPE
import signal

def set_timeout(secs, callback):
    '''
    A decorator that terminates a subprocess after timeout

    Args:
        secs: timeout seconds
        callback: callback function
    '''
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError

        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                signal.alarm(secs)
                r = func(*args, **kwargs)
                signal.alarm(0)
                return r
            except RuntimeError as e:
                callback()

        return to_do

    return wrap
def after_timeout():
    '''
    A simple callback function
    '''
    print("Time out!")
@set_timeout(5,after_timeout())
def wget_download(urls,folder):
    '''
    Downloading images from user past histories using wget

    Args:
        urls: List of image urls
        folder: The folder to save the download images

    '''
    for url in urls:
        filename = url.split('/')[-1]
        if '?' in filename:
            filename = filename.split('?')[0]
        subprocess.call(['wget',url,'-O',os.path.join(folder,filename)],stdout=PIPE,stdin=None,stderr=PIPE)

if __name__=='__main__':
    pass