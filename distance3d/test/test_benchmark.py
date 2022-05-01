import time
from distance3d.benchmark import Timer


def test_timer():
    timer = Timer()
    timer.start("1")
    time.sleep(0.05)
    timer.start("2")
    time.sleep(0.05)
    time2 = timer.stop("2")
    time1 = timer.stop("1")
    assert 0.05 <= time2 <= 0.051
    assert 0.1 <= time1 <= 0.101


def test_total_time():
    timer = Timer()
    for _ in range(5):
        timer.start("timer")
        time.sleep(0.01)
        timer.stop_and_add_to_total("timer")
    assert 0.05 <= timer.total_time_["timer"] <= 0.051
