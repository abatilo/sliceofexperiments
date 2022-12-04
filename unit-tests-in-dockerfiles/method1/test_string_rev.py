import unittest


def rev(s):
    return s[::-1]


class Test(unittest.TestCase):
    def test_string_rev(self):
        self.assertEquals(rev("abc"), "cba")
