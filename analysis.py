"""Аналіз регіональної нерівності зарплат в Україні у 2021 році."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

WAGE = "avg_monthly_wage_uah"
NATIONAL_WAGE = 14_014
COLUMNS = [
    "region", WAGE, "population", "urban_population",
    "urban_share_pct", "wage_to_ukraine_pct",
]


def load_regions(path):
    """Зчитати CSV та перевірити, чи дані придатні для порівняння."""
    data = pd.read_csv(path, encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Відсутні стовпці: {', '.join(sorted(missing))}")

    data = data[COLUMNS].copy()
    data["region"] = data["region"].astype("string").str.strip()
    names = data["region"].str.casefold()
    regions = data.loc[~names.isin(["україна", "україна в цілому"])].copy()
    if regions["region"].fillna("").eq("").any():
        raise ValueError("Є рядки без назви регіону.")
    if regions["region"].str.casefold().duplicated().any():
        raise ValueError("Є повторювані назви регіонів.")
    if len(regions) < 2:
        raise ValueError("Для порівняння потрібно щонайменше два регіони.")

    for column in COLUMNS[1:]:
        values = pd.to_numeric(regions[column], errors="coerce")
        if values.isna().any() or values.isin([float("inf"), -float("inf")]).any():
            raise ValueError(f"Пропущені або некоректні числа у стовпці {column}.")
        regions[column] = values

    if (
        regions[WAGE].le(0).any()
        or regions["population"].le(0).any()
        or regions["urban_population"].lt(0).any()
        or regions["urban_population"].gt(regions["population"]).any()
        or not regions["urban_share_pct"].between(0, 100).all()
        or regions["wage_to_ukraine_pct"].le(0).any()
    ):
        raise ValueError("Перевір зарплати, чисельність населення та відсотки.")

    urban_share = 100 * regions["urban_population"] / regions["population"]
    wage_share = 100 * regions[WAGE] / NATIONAL_WAGE
    # Допуск у 0,6 відсоткового пункту враховує округлення до цілого відсотка.
    for column, expected in [
        ("urban_share_pct", urban_share),
        ("wage_to_ukraine_pct", wage_share),
    ]:
        inconsistent = regions.loc[(regions[column] - expected).abs().gt(0.6), "region"]
        if not inconsistent.empty:
            print(f"Увага: перевір {column} для {', '.join(inconsistent)}.")

    return regions


def show_correlations(regions, label):
    """Показати кореляції Пірсона для заданої вибірки регіонів."""
    print(f"\n{label} (n = {len(regions)})")
    for column, title in [
        ("urban_share_pct", "Зарплата й частка міського населення"),
        ("population", "Зарплата й чисельність населення"),
    ]:
        if regions[WAGE].nunique() < 2 or regions[column].nunique() < 2:
            print(f"{title}: кореляцію неможливо визначити.")
        else:
            coefficient = regions[WAGE].corr(regions[column])
            print(f"{title}: r = {coefficient:.3f}")


def analyze(regions):
    print(f"Регіональна нерівність зарплат в Україні, 2021 | n = {len(regions)}")
    print("\n1. Описова статистика")
    print(regions[COLUMNS[1:]].describe().round(2).to_string())
    if len(regions) != 25:
        print("Увага: для повного порівняння очікуються 24 області та м. Київ.")

    ranking_columns = ["region", WAGE, "wage_to_ukraine_pct"]
    ranking = regions[ranking_columns].sort_values(
        [WAGE, "region"], ascending=[False, True]
    )
    print("\n2. П’ять найвищих середніх зарплат")
    print(ranking.head(5).to_string(index=False))
    print("\nП’ять найнижчих середніх зарплат")
    print(ranking.sort_values([WAGE, "region"]).head(5).to_string(index=False))

    highest = regions[WAGE].max()
    lowest = regions[WAGE].min()
    above_national = regions[WAGE].gt(NATIONAL_WAGE).sum()
    print("\n3. Розрив у зарплатах")
    print(f"Різниця між максимумом і мінімумом: {highest - lowest:,.0f} грн")
    print(f"Відношення максимуму до мінімуму: {highest / lowest:.2f}")
    print(f"Медіана середніх регіональних зарплат: {regions[WAGE].median():,.2f} грн")
    print(f"Регіонів вище показника України ({NATIONAL_WAGE:,} грн): {above_national}")

    print("\n4. Зв’язок зарплати з населенням та урбанізацією")
    show_correlations(regions, "Усі регіони")
    names = regions["region"].str.casefold().str.replace(r"\s+", "", regex=True)
    is_capital = names.isin(["київ", "м.київ", "мкиїв", "містокиїв"])
    if is_capital.any() and (~is_capital).sum() >= 2:
        show_correlations(regions.loc[~is_capital], "Без міста Києва")
    print("\nКореляції описують лінійні зв’язки між регіональними показниками.")


def save_charts(regions, output_dir):
    """Зберегти рейтинг зарплат і дві діаграми розсіювання."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = regions["region"].str.casefold().str.replace(r"\s+", "", regex=True)
    capital = names.isin(["київ", "м.київ", "мкиїв", "містокиїв"])
    chart_data = regions.assign(is_capital=capital)
    ranking = chart_data.sort_values(WAGE)
    colors = ranking["is_capital"].map({True: "#c05b24", False: "#256783"})

    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    ax.barh(ranking["region"], ranking[WAGE], color=colors)
    ax.axvline(NATIONAL_WAGE, color="#52525b", linestyle="--",
               label="Україна — 14 014 грн")
    ax.set(title="Середня зарплата за регіонами України, 2021",
           xlabel="Середньомісячна зарплата, грн")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "wage_by_region.png", dpi=160, facecolor="white")
    plt.close(fig)

    for column, title, xlabel, filename in [
        ("urban_share_pct", "Зарплата й урбанізація, 2021",
         "Частка міського населення, %", "wage_vs_urbanization.png"),
        ("population", "Зарплата й чисельність населення, 2021",
         "Чисельність населення (в одиницях CSV)", "wage_vs_population.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
        oblasts = chart_data.loc[~capital]
        ax.scatter(oblasts[column], oblasts[WAGE], color="#256783", alpha=0.85,
                   label="Області")
        if capital.any():
            city = chart_data.loc[capital]
            ax.scatter(city[column], city[WAGE], color="#c05b24", s=70, label="м. Київ")
            ax.legend()
        ax.set(title=title, xlabel=xlabel, ylabel="Середньомісячна зарплата, грн")
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        fig.savefig(output_dir / filename, dpi=160, facecolor="white")
        plt.close(fig)
    print(f"\nГрафіки збережено в {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "data" / "regional_economic_data_2021.csv",
        help="Шлях до CSV. За замовчуванням — data/ поруч зі скриптом.",
    )
    parser.add_argument("--charts-dir", type=Path, default=Path(__file__).resolve().parent / "charts")
    args = parser.parse_args()
    try:
        regions = load_regions(args.csv)
        analyze(regions)
        save_charts(regions, args.charts_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Помилка аналізу: {error}\n")


if __name__ == "__main__":
    main()
