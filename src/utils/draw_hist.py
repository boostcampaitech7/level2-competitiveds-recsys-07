import matplotlib.pyplot as plt
import numpy as np


# 히스토그램 그리기
def plot_histogram_with_mean(df, column_name, k=5):
    plt.figure(figsize=(10, 6))

    n, bins, patches = plt.hist(
        df[column_name], bins=np.arange(df[column_name].min(), df[column_name].max() + k, k), edgecolor="black"
    )

    plt.title(f"Distribution of {column_name.capitalize()}", fontsize=20, fontweight="bold")
    plt.xlabel(column_name.capitalize(), fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.5)

    mean_value = df[column_name].mean()
    plt.axvline(
        mean_value,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Mean {column_name.capitalize()}: {mean_value:.2f}",
    )

    plt.legend(fontsize=10)

    # x축 설정
    plt.xlim(df[column_name].min(), df[column_name].max())
    plt.xticks(np.arange(df[column_name].min(), df[column_name].max() + 5, 5), fontsize=10)  # 5단위로 눈금 표시

    # y축 설정
    max_frequency = n.max()
    plt.ylim(0, max_frequency * 1.1)
    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: format(int(x), ","))
    )  # y축의 눈금 레이블을 포맷팅

    plt.tick_params(axis="both", which="major", labelsize=7)

    # 각 막대 위에 값 추가
    for i in range(len(patches)):
        height = patches[i].get_height()
        plt.text(
            patches[i].get_x() + patches[i].get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()
