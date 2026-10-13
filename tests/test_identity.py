from reidfm.data.identity import parse_last_filename, visual_train_pid


def test_visual_train_pid_splits_known_clothes():
    assert visual_train_pid(42, 3) == "42c3"
    assert visual_train_pid("42", "3") == "42c3"
    assert visual_train_pid(42, 0) == "42"
    assert visual_train_pid(42, "0") == "42"
    assert visual_train_pid(42, None) == "42"
    assert visual_train_pid(42, "") == "42"


def test_parse_last_filename():
    pid, clothes = parse_last_filename("1001_2_3_4_5_6.jpg")
    assert pid == "1001"
    assert clothes == "6"
    assert visual_train_pid(pid, clothes) == "1001c6"
