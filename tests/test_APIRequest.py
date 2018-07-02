import json
import sys
sys.path.append('../')
import gitfeatures as gf


class TestAPIRequest(object):
    """Test the Github API request function."""

    def setup(self):
        self.url = 'https://api.github.com/repos/silburt/DeepMoon'

    def test_APIRequest(self):
        r = gf.get_request(self.url)
        forks = 0
        if r.ok:
            contents = json.loads(r.text or r.content)
            forks = contents['forks_count']
        assert forks == 5

    def test_RequestFailure(self):
        r = gf.get_request(self.url, timeout=1e-4)
        assert r is None
