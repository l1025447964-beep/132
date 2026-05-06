# data_analysis_agent.py
# 用法：
# 1. pip install pandas numpy matplotlib
# 2. python data_analysis_agent.py your_data.csv

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class DataAnalysisAgent:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.report_lines = []
        self.output_dir = "agent_output"

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        """读取 CSV 数据"""
        self.df = pd.read_csv(self.file_path)

        self.report_lines.append("# AI 数据分析 Agent 报告")
        self.report_lines.append("")
        self.report_lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report_lines.append(f"数据文件：`{self.file_path}`")
        self.report_lines.append("")

    def basic_overview(self):
        """基础数据概览"""
        rows, cols = self.df.shape

        self.report_lines.append("## 1. 数据概览")
        self.report_lines.append("")
        self.report_lines.append(f"- 数据行数：{rows}")
        self.report_lines.append(f"- 数据列数：{cols}")
        self.report_lines.append(f"- 字段列表：{', '.join(self.df.columns)}")
        self.report_lines.append("")

        self.report_lines.append("### 字段类型")
        self.report_lines.append("")
        for col, dtype in self.df.dtypes.items():
            self.report_lines.append(f"- `{col}`：{dtype}")
        self.report_lines.append("")

    def missing_value_analysis(self):
        """缺失值分析"""
        self.report_lines.append("## 2. 缺失值分析")
        self.report_lines.append("")

        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            self.report_lines.append("未发现明显缺失值。")
        else:
            for col, count in missing.items():
                ratio = count / len(self.df) * 100
                self.report_lines.append(
                    f"- `{col}` 缺失 {count} 条，占比 {ratio:.2f}%"
                )

        self.report_lines.append("")

    def numeric_summary(self):
        """数值字段统计"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        self.report_lines.append("## 3. 数值字段统计")
        self.report_lines.append("")

        if not numeric_cols:
            self.report_lines.append("未发现数值型字段。")
            self.report_lines.append("")
            return

        summary = self.df[numeric_cols].describe().T

        self.report_lines.append("| 字段 | 均值 | 中位数 | 最小值 | 最大值 | 标准差 |")
        self.report_lines.append("|---|---:|---:|---:|---:|---:|")

        for col in numeric_cols:
            self.report_lines.append(
                f"| {col} | "
                f"{self.df[col].mean():.2f} | "
                f"{self.df[col].median():.2f} | "
                f"{self.df[col].min():.2f} | "
                f"{self.df[col].max():.2f} | "
                f"{self.df[col].std():.2f} |"
            )

        self.report_lines.append("")

    def anomaly_detection(self):
        """基于 IQR 的异常值检测"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        self.report_lines.append("## 4. 异常值检测")
        self.report_lines.append("")

        if not numeric_cols:
            self.report_lines.append("未发现可检测异常值的数值字段。")
            self.report_lines.append("")
            return

        found = False

        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            anomalies = self.df[(self.df[col] < lower) | (self.df[col] > upper)]

            if len(anomalies) > 0:
                found = True
                ratio = len(anomalies) / len(self.df) * 100
                self.report_lines.append(
                    f"- `{col}` 发现 {len(anomalies)} 条异常值，占比 {ratio:.2f}%"
                )

        if not found:
            self.report_lines.append("未发现明显异常值。")

        self.report_lines.append("")

    def correlation_analysis(self):
        """相关性分析"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        self.report_lines.append("## 5. 相关性分析")
        self.report_lines.append("")

        if len(numeric_cols) < 2:
            self.report_lines.append("数值型字段少于 2 个，无法进行相关性分析。")
            self.report_lines.append("")
            return

        corr = self.df[numeric_cols].corr()

        strong_pairs = []

        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                col_a = numeric_cols[i]
                col_b = numeric_cols[j]
                value = corr.loc[col_a, col_b]

                if abs(value) >= 0.6:
                    strong_pairs.append((col_a, col_b, value))

        if not strong_pairs:
            self.report_lines.append("未发现强相关字段。")
        else:
            for col_a, col_b, value in strong_pairs:
                direction = "正相关" if value > 0 else "负相关"
                self.report_lines.append(
                    f"- `{col_a}` 与 `{col_b}` 存在较强{direction}，相关系数为 {value:.2f}"
                )

        self.report_lines.append("")

    def trend_analysis(self):
        """尝试识别时间字段并分析趋势"""
        self.report_lines.append("## 6. 趋势分析")
        self.report_lines.append("")

        date_col = None

        for col in self.df.columns:
            try:
                parsed = pd.to_datetime(self.df[col], errors="coerce")
                valid_ratio = parsed.notnull().mean()

                if valid_ratio > 0.7:
                    date_col = col
                    self.df[col] = parsed
                    break
            except Exception:
                continue

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if date_col is None or not numeric_cols:
            self.report_lines.append("未识别到足够清晰的时间字段或数值字段，暂不生成趋势分析。")
            self.report_lines.append("")
            return

        self.report_lines.append(f"识别到时间字段：`{date_col}`")
        self.report_lines.append("")

        df_sorted = self.df.sort_values(date_col)

        for col in numeric_cols[:3]:
            trend_file = os.path.join(self.output_dir, f"trend_{col}.png")

            plt.figure(figsize=(10, 5))
            plt.plot(df_sorted[date_col], df_sorted[col])
            plt.title(f"{col} 趋势图")
            plt.xlabel(date_col)
            plt.ylabel(col)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(trend_file)
            plt.close()

            start_value = df_sorted[col].iloc[0]
            end_value = df_sorted[col].iloc[-1]

            if end_value > start_value:
                trend = "上升"
            elif end_value < start_value:
                trend = "下降"
            else:
                trend = "基本持平"

            self.report_lines.append(
                f"- `{col}` 从首期到末期整体呈现 **{trend}** 趋势。趋势图已生成：`{trend_file}`"
            )

        self.report_lines.append("")

    def generate_recommendations(self):
        """生成业务建议"""
        self.report_lines.append("## 7. 初步结论与建议")
        self.report_lines.append("")
        self.report_lines.append("- 优先关注缺失值较高的字段，确认是否会影响后续分析口径。")
        self.report_lines.append("- 对异常值较多的指标，需要判断是业务波动、数据采集问题，还是真实风险信号。")
        self.report_lines.append("- 对强相关字段可进一步分析因果关系，避免只根据相关性直接下结论。")
        self.report_lines.append("- 如果存在时间趋势，建议结合活动、版本发布、渠道变化等业务事件做归因分析。")
        self.report_lines.append("- 后续可以接入数据库、BI 系统或大模型接口，实现自动取数、自动解释和自动周报生成。")
        self.report_lines.append("")

    def save_report(self):
        """保存 Markdown 报告"""
        report_path = os.path.join(self.output_dir, "data_analysis_report.md")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.report_lines))

        return report_path

    def run(self):
        """Agent 主流程"""
        self.load_data()
        self.basic_overview()
        self.missing_value_analysis()
        self.numeric_summary()
        self.anomaly_detection()
        self.correlation_analysis()
        self.trend_analysis()
        self.generate_recommendations()

        report_path = self.save_report()
        print(f"分析完成，报告已生成：{report_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供 CSV 文件路径，例如：")
        print("python data_analysis_agent.py sales_data.csv")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        sys.exit(1)

    agent = DataAnalysisAgent(file_path)
    agent.run()