from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(data_path: str = "data/traffic_data.csv", output_dir: str = "artifacts/eda") -> None:
    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.day_name()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df.groupby("hour", as_index=False)["vehicle_count"].mean(), x="hour", y="vehicle_count")
    plt.title("Average Vehicle Count by Hour")
    plt.tight_layout()
    plt.savefig(out / "hourly_vehicle_trend.png")
    plt.close()

    pivot = df.pivot_table(index="zone_id", columns="hour", values="traffic_speed", aggfunc="mean")
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot, cmap="coolwarm", center=pivot.mean().mean())
    plt.title("Congestion Heatmap by Zone and Hour")
    plt.tight_layout()
    plt.savefig(out / "zone_hour_heatmap.png")
    plt.close()

    corr = df[["traffic_speed", "vehicle_count", "distance_km", "travel_time_min"]].corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr, annot=True, cmap="Blues")
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig(out / "correlation_heatmap.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="weather", y="travel_time_min")
    plt.title("Travel Time Distribution by Weather")
    plt.tight_layout()
    plt.savefig(out / "weather_vs_travel_time.png")
    plt.close()


if __name__ == "__main__":
    run_eda()
    print("EDA charts saved in artifacts/eda.")
