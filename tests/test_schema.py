from pandera import Column, DataFrameSchema, Check
from pathlib import Path
from src.features import load_raw_data, NUM_COLS, CAT_COLS

raw_schema = DataFrameSchema({
    "tenure": Column(int, checks=[
        Check.greater_than_or_equal_to(0),
        Check.less_than_or_equal_to(200),
    ]),
    "MonthlyCharges": Column(float, checks=[
        Check.greater_than_or_equal_to(0),
    ]),
    "TotalCharges": Column(float, checks=[
        Check.greater_than_or_equal_to(0),
    ], nullable=True),
    "Churn": Column(str, checks=[
        Check.isin(["Yes", "No"]),
    ]),
    "Contract": Column(str, checks=[
        Check.isin(["Month-to-month", "One year", "Two year"]),
    ]),
    "InternetService": Column(str, checks=[
        Check.isin(["DSL", "Fiber optic", "No"]),
    ]),
    "gender": Column(str, checks=[
        Check.isin(["Male", "Female"]),
    ]),
})


def test_raw_data_schema():
    """Dataset bruto deve respeitar o schema definido."""
    df = load_raw_data(Path("data/raw/telco_churn.csv"))

    # Adicionar Churn de volta para validação (load_raw_data não remove)
    validated = raw_schema.validate(df)
    assert validated is not None


def test_no_duplicates():
    """Dataset não deve ter linhas duplicadas."""
    df = load_raw_data(Path("data/raw/telco_churn.csv"))
    assert df.duplicated().sum() == 0


def test_numeric_columns_present():
    """Colunas numéricas obrigatórias devem existir."""
    df = load_raw_data(Path("data/raw/telco_churn.csv"))
    for col in NUM_COLS:
        assert col in df.columns, f"Coluna {col} não encontrada"


def test_categorical_columns_present():
    """Colunas categóricas obrigatórias devem existir."""
    df = load_raw_data(Path("data/raw/telco_churn.csv"))
    for col in CAT_COLS:
        assert col in df.columns, f"Coluna {col} não encontrada"


def test_target_binary():
    """Target deve ter apenas dois valores: Yes e No."""
    df = load_raw_data(Path("data/raw/telco_churn.csv"))
    assert set(df["Churn"].unique()) == {"Yes", "No"}