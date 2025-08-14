if __name__ == "__main__": import __config__
import argparse
from SlidingWindowExperiment.StatisticalAnalysis import StatisticalAnalysis
import os

def add_column(df, column_name, callback=lambda i,row: ""):
    for i, row in df.iterrows():
        # print(i,row)
        df[i,column_name] = callback(i, row)
    return df

def main(directories, output):
    sa = StatisticalAnalysis(directories)
    print(sa.concat_df)
    all_together = sa.get_report_ir_ttest_metric_df_v2()
    all_together.to_csv(output, sep=",", decimal='.', index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Results Analysis',
        description='Program reads a given directory and search for .csv files containing \'concat\' substring is the name '
                    'the combines all those files into one dataframe and makes statistical analysis including '
                    'improvement ratio calculation and Welch\'s test. The final file is saved in output.',
        epilog='example: python examples/experiments/problem_analysis_results.py cm/all cm/sa.csv')

    parser.add_argument('inputdir', help="Directory that contains output .csvs from experimental script")
    parser.add_argument('outputfile', help="Name of the output file. It should be csv")
    args = parser.parse_args()
    parser.print_help()

    main([args.inputdir], args.outputfile)