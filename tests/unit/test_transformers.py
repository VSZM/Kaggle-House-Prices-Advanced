from transformers import *
import pytest

@pytest.fixture(scope = 'module')
def df():
    df = pd.DataFrame()
    df['A'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df['B'] = ['a', 'a', 'b', 'a', 'c', 'c', 'd', 'a', 'b', 'a']
    return df

@pytest.fixture(scope = 'module')
def df_na():
    df = pd.DataFrame()
    df['A'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df['B'] = ['a', 'a', 'b', 'a', None, 'c', None, 'a', 'b', 'a']
    return df


def assert_array_equals(expected, actual):
    assert all([a == b for a, b in zip(expected, actual)]), f'Expected: |{expected}| != Actual: |{actual}|'

def test_drop(df):
    transformer = FeatureDropTransformer(['A'])
    result = transformer.fit_transform(df)
    assert 'A' not in result.columns
    assert 'B' in result.columns

def test_clip(df):
    transformer = ClipTransformer({'A': (4,7)})
    result = transformer.fit_transform(df)
    assert_array_equals([4, 4, 4, 4, 5, 6, 7, 7, 7, 7], result.A)

def test_typo(df):
    transformer = FixTypoTransformer({'B': {'d': 'b'}})
    result = transformer.fit_transform(df)
    assert_array_equals(['a', 'a', 'b', 'a', 'c', 'c', 'b', 'a', 'b', 'a'], result.B)

def test_fillna(df_na):
    transformer = FillNATransformer(['B'], 'x')
    result = transformer.fit_transform(df_na)
    assert_array_equals(['a', 'a', 'b', 'a', 'x', 'c', 'x', 'a', 'b', 'a'], result.B)