import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from typing import List
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from cluster_engine import ClusteringExperimentResult


class ReportVisualizer:
    """Генератор иллюстративных графиков"""

    def __init__(self, output_dir: str = "output_figures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
        plt.rcParams["axes.unicode_minus"] = False

    def plot_dendrogram(self, res: ClusteringExperimentResult, labels: List[str], cut_k: int, filename: str,) -> Path:
        """Строит дендрограмму агломерации с линией порога отсечения для K кластеров"""
        z_link = res.linkage_matrix
        n = len(labels)
        h_low = z_link[n - cut_k - 1, 2]
        h_high = z_link[n - cut_k, 2]
        threshold = 0.5 * (h_low + h_high)

        fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)
        dendrogram(
            z_link,
            labels=labels,
            orientation="left",
            color_threshold=threshold,
            above_threshold_color="#4A5568",
            leaf_font_size=10,
            ax=ax,
        )
        ax.axvline(
            x=threshold,
            color="#C53030",
            linestyle="--",
            linewidth=1.6,
            label=f"Порог сечения для K = {cut_k} (h = {threshold:.2f})",
        )
        ax.set_title(
            f"Дендрограмма кластеризации регионов ПФО ({res.method_title}, {res.year} г.)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Евклидово расстояние слияния ", fontsize=10.5)
        ax.grid(axis="x", linestyle=":", alpha=0.6)
        ax.legend(loc="lower right", frameon=True, fontsize=9.5)
        plt.tight_layout()

        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_validation_curves(self, metrics_all: pd.DataFrame, filename: str) -> Path:
        """Строит графики зависимости S(K) и DB(K) от числа кластеров за 2022 и 2023"""
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), dpi=300)
        years = ["2022", "2023"]
        styles = {
            "Центроидный метод": {"color": "#2B6CB0", "marker": "s", "linestyle": "--"},
            "Метод Уорда": {"color": "#C53030", "marker": "o", "linestyle": "-"},
        }

        for col_idx, yr in enumerate(years):
            sub = metrics_all[metrics_all["Год"] == yr]
            ax_sil = axes[0, col_idx]
            ax_db = axes[1, col_idx]

            for m_title, st in styles.items():
                m_sub = sub[sub["Метод"] == m_title]
                ax_sil.plot(
                    m_sub["Число кластеров K"],
                    m_sub["Коэффициент силуэта S(K)"],
                    label=m_title,
                    linewidth=2.0,
                    markersize=6,
                    **st,
                )
                ax_db.plot(
                    m_sub["Число кластеров K"],
                    m_sub["Индекс Дэвиса-Болдина DB(K)"],
                    label=m_title,
                    linewidth=2.0,
                    markersize=6,
                    **st,
                )

            ax_sil.set_title(
                f"Коэффициент силуэта S(K), {yr} г (max - оптимум)",
                fontsize=10.5,
                fontweight="bold",
            )
            ax_sil.set_ylabel("Значение S(K)", fontsize=9.5)
            ax_sil.set_xticks([2, 3, 4, 5, 6])
            ax_sil.set_ylim(0.20, 0.65)
            ax_sil.grid(True, linestyle=":", alpha=0.6)
            ax_sil.legend(fontsize=9)

            ax_db.set_title(
                f"Индекс Дэвиса - Болдина DB(K), {yr} г (min - оптимум)",
                fontsize=10.5,
                fontweight="bold",
            )
            ax_db.set_xlabel("Число кластеров (K)", fontsize=9.5)
            ax_db.set_ylabel("Значение DB(K)", fontsize=9.5)
            ax_db.set_xticks([2, 3, 4, 5, 6])
            ax_db.set_ylim(0.25, 0.95)
            ax_db.grid(True, linestyle=":", alpha=0.6)
            ax_db.legend(fontsize=9)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_architecture_diagram(self, filename: str) -> Path:
        """Генерирует структурную схему модулей и потоков данных"""
        fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 52)
        ax.axis("off")

        def draw_box(x, y, w, h, title, body, facecolor="#EDF2F7", edgecolor="#2B6CB0"):
            box = patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.6,rounding_size=1.2",
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=1.5,
            )
            ax.add_patch(box)
            ax.text(
                x + w / 2,
                y + h - 2.6,
                title,
                ha="center",
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color="#1A202C",
            )
            ax.text(
                x + w / 2,
                y + h / 2 - 1.2,
                body,
                ha="center",
                va="center",
                fontsize=8.3,
                color="#2D3748",
            )

        draw_box(
            2,
            18,
            19,
            16,
            "Входные данные",
            "Файл Росстата (.xlsx)\nЛисты: 2022, 2023\nМатрицы X (14 x 7)",
            facecolor="#EBF8FF",
            edgecolor="#2B6CB0",
        )
        draw_box(
            26,
            30,
            22,
            16,
            "1. data_loader.py",
            "SocioEconomicDataLoader\nПроверка NaN и X > 0\nz-стандартизация (ddof=1)",
            facecolor="#F7FAFC",
            edgecolor="#2C5282",
        )
        draw_box(
            26,
            6,
            22,
            16,
            "2. cluster_engine.py",
            "HierarchicalClusterEngine\nМатрица D (14 x 14)\nЦентроидный метод и Уорд",
            facecolor="#F7FAFC",
            edgecolor="#2C5282",
        )
        draw_box(
            54,
            18,
            22,
            16,
            "3. Валидация метрик",
            "Силуэт S(K), Дэвис - Болдин\nКофенетическая корреляция\nДетектор инверсий и ARI",
            facecolor="#F7FAFC",
            edgecolor="#2C5282",
        )
        draw_box(
            81,
            18,
            17,
            16,
            "4. visualizer.py",
            "ReportVisualizer\nДендрограммы (.png)\nКривые и отчет (.xlsx)",
            facecolor="#F0FFF4",
            edgecolor="#276749",
        )

        arrow_kw = dict(
            arrowstyle="-|>,head_length=0.6,head_width=0.35", lw=1.5, color="#2D3748"
        )
        ax.annotate("", xy=(25.5, 38), xytext=(21.5, 29), arrowprops=arrow_kw)
        ax.annotate("", xy=(37, 22.8), xytext=(37, 29.4), arrowprops=arrow_kw)
        ax.annotate("", xy=(53.5, 24), xytext=(48.6, 15), arrowprops=arrow_kw)
        ax.annotate("", xy=(80.5, 26), xytext=(76.6, 26), arrowprops=arrow_kw)

        ax.set_title(
            "Архитектура программного комплекса кластерного анализа регионов ПФО",
            fontsize=11.5,
            fontweight="bold",
            pad=6,
        )
        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path