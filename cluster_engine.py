from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (adjusted_rand_score, davies_bouldin_score, silhouette_samples, silhouette_score,)

@dataclass
class ClusteringExperimentResult:
    """Контейнер результатов для одного года и метода"""

    year: str
    method_code: str
    method_title: str
    distance_matrix: pd.DataFrame
    linkage_matrix: np.ndarray
    cophenetic_corr: float
    inversions: List[Tuple[int, float, float]]
    metrics_table: pd.DataFrame
    partitions: Dict[int, np.ndarray]
    sample_silhouettes: Dict[int, np.ndarray]


class HierarchicalClusterEngine:
    """Реализация восходящей иерархической кластеризации"""

    SUPPORTED_METHODS: Dict[str, str] = {
        "centroid": "Центроидный метод",
        "ward": "Метод Уорда",
    }

    def __init__(self, k_min: int = 2, k_max: int = 6) -> None:
        if k_min < 2 or k_max <= k_min:
            raise ValueError("Некорректный диапазон числа кластеров [k_min, k_max]")
        self.k_min = k_min
        self.k_max = k_max

    @staticmethod
    def compute_distance_matrix(z_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Вычисляет конденсированный вектор и квадратную матрицу евклидовых расстояний"""

        dist_condensed = pdist(z_df.values, metric="euclidean")
        dist_square = pd.DataFrame(
            squareform(dist_condensed),
            index=z_df.index,
            columns=z_df.index,
        )
        return dist_condensed, dist_square

    @staticmethod
    def detect_inversions(linkage_matrix: np.ndarray) -> List[Tuple[int, float, float]]:
        """Проверяет условие монотонности высот слияния на дендрограмме"""

        heights = linkage_matrix[:, 2]
        inversions: List[Tuple[int, float, float]] = []
        for step in range(1, len(heights)):
            if heights[step] < heights[step - 1]:
                inversions.append(
                    (step + 1, float(heights[step - 1]), float(heights[step]))
                )
        return inversions

    def run_experiment(self, year: str, z_df: pd.DataFrame, method_code: str) -> ClusteringExperimentResult:
        """Выполняет агломерацию и рассчитывает метрики"""

        if method_code not in self.SUPPORTED_METHODS:
            raise ValueError(f"Неподдерживаемый метод кластеризации: {method_code}")

        dist_condensed, dist_square = self.compute_distance_matrix(z_df)
        z_link = linkage(z_df.values, method=method_code, metric="euclidean")
        coph_r, _ = cophenet(z_link, dist_condensed)
        inversions = self.detect_inversions(z_link)

        rows = []
        partitions: Dict[int, np.ndarray] = {}
        sample_silhouettes: Dict[int, np.ndarray] = {}

        for k in range(self.k_min, self.k_max + 1):
            labels = fcluster(z_link, t=k, criterion="maxclust")
            sil_avg = float(silhouette_score(z_df.values, labels, metric="euclidean"))
            db_val = float(davies_bouldin_score(z_df.values, labels))
            sil_samples = silhouette_samples(z_df.values, labels, metric="euclidean")

            sizes = np.bincount(labels)[1:]
            partitions[k] = labels
            sample_silhouettes[k] = sil_samples

            rows.append(
                {
                    "Год": year,
                    "Метод": self.SUPPORTED_METHODS[method_code],
                    "Число кластеров K": k,
                    "Коэффициент силуэта S(K)": round(sil_avg, 4),
                    "Индекс Дэвиса-Болдина DB(K)": round(db_val, 4),
                    "Наполненность кластеров": " + ".join(map(str, sizes)),
                }
            )

        return ClusteringExperimentResult(
            year=year,
            method_code=method_code,
            method_title=self.SUPPORTED_METHODS[method_code],
            distance_matrix=dist_square,
            linkage_matrix=z_link,
            cophenetic_corr=float(coph_r),
            inversions=inversions,
            metrics_table=pd.DataFrame(rows),
            partitions=partitions,
            sample_silhouettes=sample_silhouettes,
        )

    @staticmethod
    def evaluate_interyear_stability( res_2022: ClusteringExperimentResult, res_2023: ClusteringExperimentResult, k_values: List[int],) -> pd.DataFrame:
        """Оценивает межгодовую устойчивость кластеров по скорректированному индексу Рэнда"""

        records = []
        for k in k_values:
            ari = adjusted_rand_score(res_2022.partitions[k], res_2023.partitions[k])
            records.append(
                {
                    "Метод": res_2022.method_title,
                    "Число кластеров K": k,
                    "Индекс устойчивости (2022 vs 2023)": round(float(ari), 4),
                }
            )
        return pd.DataFrame(records)