import argparse
import time
from typing import Dict, Tuple
import pandas as pd

from cluster_engine import ClusteringExperimentResult, HierarchicalClusterEngine
from data_loader import SocioEconomicDataLoader
from visualizer import ReportVisualizer


def parse_args() -> argparse.Namespace:
    """Разбор аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Программный комплекс иерархической кластеризации регионов ПФО"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data.xlsx",
        help="Путь к входному Excel-файлу с данными Росстата",
    )
    parser.add_argument(
        "--output-xlsx",
        type=str,
        default="Результаты_кластеризации.xlsx",
        help="Имя выходного файла отчета Excel",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="output_figures",
        help="Каталог для сохранения графиков",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа"""
    t_start = time.perf_counter()
    args = parse_args()

    loader = SocioEconomicDataLoader(args.input)
    engine = HierarchicalClusterEngine(k_min=2, k_max=6)
    visualizer = ReportVisualizer(output_dir=args.figures_dir)

    all_metrics = []
    results_by_key: Dict[Tuple[str, str], ClusteringExperimentResult] = {}
    z_matrices: Dict[str, pd.DataFrame] = {}

    for year in ["2022", "2023"]:
        df_raw = loader.load_and_validate(year)
        df_z, _, _ = loader.standardize(df_raw)
        z_matrices[year] = df_z

        for method_code in ["centroid", "ward"]:
            res = engine.run_experiment(year=year, z_df=df_z, method_code=method_code)
            results_by_key[(year, method_code)] = res
            all_metrics.append(res.metrics_table)
            print(
                f"[{year} | {res.method_title}] "
                f"r_coph = {res.cophenetic_corr:.4f}, инверсий = {len(res.inversions)}"
            )

    metrics_summary = pd.concat(all_metrics, ignore_index=True)
    stability_ward = engine.evaluate_interyear_stability(
        results_by_key[("2022", "ward")],
        results_by_key[("2023", "ward")],
        k_values=[2, 3, 4],
    )
    stability_centroid = engine.evaluate_interyear_stability(
        results_by_key[("2022", "centroid")],
        results_by_key[("2023", "centroid")],
        k_values=[2, 3, 4],
    )
    stability_all = pd.concat([stability_centroid, stability_ward], ignore_index=True)

    # Генерация графиков
    visualizer.plot_architecture_diagram("Рисунок_3_1_Архитектура_ПО.png")
    visualizer.plot_dendrogram(
        results_by_key[("2022", "centroid")],
        labels=list(z_matrices["2022"].index),
        cut_k=3,
        filename="Рисунок_3_2_Дендрограмма_центроидный_2022.png",
    )
    visualizer.plot_dendrogram(
        results_by_key[("2022", "ward")],
        labels=list(z_matrices["2022"].index),
        cut_k=3,
        filename="Рисунок_3_3_Дендрограмма_Уорда_2022.png",
    )
    visualizer.plot_dendrogram(
        results_by_key[("2023", "centroid")],
        labels=list(z_matrices["2023"].index),
        cut_k=3,
        filename="Рисунок_3_4_Дендрограмма_центроидный_2023.png",
    )
    visualizer.plot_dendrogram(
        results_by_key[("2023", "ward")],
        labels=list(z_matrices["2023"].index),
        cut_k=3,
        filename="Рисунок_3_5_Дендрограмма_Уорда_2023.png",
    )
    visualizer.plot_validation_curves(
        metrics_summary, "Рисунок_3_6_Метрики_качества_2022_2023.png"
    )

    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        metrics_summary.to_excel(writer, sheet_name="Метрики_качества", index=False)
        stability_all.to_excel(writer, sheet_name="Устойчивость_ARI", index=False)
        z_matrices["2022"].to_excel(writer, sheet_name="Z_матрица_2022")
        z_matrices["2023"].to_excel(writer, sheet_name="Z_матрица_2023")

    elapsed = time.perf_counter() - t_start
    print(f"Вычислительный эксперимент завершен за {elapsed:.3f} с")


if __name__ == "__main__":
    main()