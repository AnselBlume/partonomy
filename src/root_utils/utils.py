import time

def get_timestr():
    return time.strftime("%Y_%m_%d-%H_%M_%S", time.localtime())