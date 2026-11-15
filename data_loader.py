from pathlib import Path
from typing import Tuple
import pandas as pd


class SocioEconomicDataLoader:
    """Загружает и валидирует матрицы наблюдений"""

    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Файл исходных данных не найден по указанному пути: {self.filepath}"
            )

    def load_and_validate(self, sheet_name: str) -> pd.DataFrame:
        """Считывает таблицу временного среза и проверяет корректность значений"""

        try:
            df = pd.read_excel(self.filepath, sheet_name=sheet_name, index_col=0)
        except ValueError as exc:
            raise ValueError(
                f"Лист '{sheet_name}' отсутствует в книге '{self.filepath.name}'"
            ) from exc

        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()

        if df.empty:
            raise ValueError(f"Матрица наблюдений на листе '{sheet_name}' пуста")
        if df.isnull().any().any():
            raise ValueError(
                f"На листе '{sheet_name}' обнаружены пропущенные значения"
            )
        if (df <= 0).any().any():
            raise ValueError(
                f"На листе '{sheet_name}' обнаружены нулевые или отрицательные значения ВРП"
            )
        return df

    @staticmethod
    def standardize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Выполняет одномерную z-стандартизацию признаков"""
        
        means = df.mean(axis=0)
        stds = df.std(axis=0, ddof=1)
        if (stds == 0).any():
            zero_cols = list(stds[stds == 0].index)
            raise ZeroDivisionError(
                f"Нулевое стандартное отклонение по признакам: {zero_cols}"
            )
        z_df = (df - means) / stds
        return z_df, means, stds