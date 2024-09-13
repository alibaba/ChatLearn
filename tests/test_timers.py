import time

from chatlearn.utils.timer import Timers


timers = Timers()
timers("aaa").start()
time.sleep(1)
sec = timers("aaa").elapsed()
assert sec < 1.1 and sec > 1
print(sec)
time.sleep(1)
sec = timers("aaa").elapsed()
assert sec < 1.1 and sec > 1
print(sec)

timers("aaa").stop()
timers("aaa").start()
time.sleep(0.5)
timers("aaa").stop()
time.sleep(0.5)
timers("aaa").start()
time.sleep(0.5)
timers("aaa").stop()

sec = timers("aaa").elapsed()
assert sec > 1 and sec < 1.1
