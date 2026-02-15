from src.preprocess import preprocess

def test_preprocess():
    df = preprocess("data/sample.csv")
    assert not df.empty
    assert df.isnull().sum().sum() == 0
