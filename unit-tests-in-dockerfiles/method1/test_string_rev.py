import unittest

from main import rev


class Test(unittest.TestCase):
    def test_string_rev(self):
        self.assertEquals(rev("abc"), "cba")
