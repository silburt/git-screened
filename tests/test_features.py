import sys
sys.path.append('../')
import gitfeatures as gf
import gitscraper as gs


class TestFeatures(object):
    """Test features scraped from each Github repo."""

    def setup(self):
        self.text = open('sample_script.py', 'r').read()

    def test_codelines(self):
        text = self.text
        code_lines = len(text.split('\n'))
        assert code_lines == 110

    def test_docstrings(self):
        text = self.text
        n_docs = gf.get_comments(text, '"""', '"""')
        assert n_docs == 51

    def test_comments(self):
        text = self.text
        n_comments = gf.get_comments(text, '#', '\n')
        assert n_comments == 14

    def test_Pep8(self):
        text = self.text
        GP = gs.Github_Profile()
        gf.get_pep8_errs(text, GP)
        assert GP.pep8['E1'] == 1
        assert GP.pep8['E2'] == 1
        assert GP.pep8['E5'] == 1
        assert GP.pep8['E7'] == 1
        assert GP.pep8['W2'] == 3


class TestReadme(object):
    """Test readme length feature."""

    def test_readme(self):
        GP = gs.Github_Profile()
        url = 'https://api.github.com/repos/silburt/DeepMoon/contents'
        gf.get_readme_length(url, GP)
        assert GP.readme_lines == 99
