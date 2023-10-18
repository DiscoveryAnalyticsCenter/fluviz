#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# vim: ts=4 sts=4 sw=4 tw=88 sta et
"""%prog [options]
Python source code - @todo
"""

from pathlib import Path
import pandas as pd


__author__ = "Patrick Butler"
__email__ = "pabutler@vt.edu"
__version__ = "0.0.1"


locs = pd.read_csv(Path("docs") / "locations.csv", dtype={"location": str})
loc_lookup = dict(locs[["location_name", "location"]].to_numpy())


def get_models(fcast_dir: Path):
    models = [d.name for d in fcast_dir.iterdir() if d.is_dir()]
    return models


def process_model_dir(model_dir: Path):
    model_name = model_dir.name
    all_results = pd.DataFrame()
    for forecast in model_dir.iterdir():
        if forecast.suffix != ".csv":
            continue
        df_pred = pd.read_csv(forecast, dtype={"location": str})
        results = get_model_results(df_pred, model_name)
        all_results = pd.concat([all_results, results], axis=0)
    all_results.reset_index(drop=True, inplace=True)
    return all_results


def get_model_results(df, model_name):
    model_preds = pd.DataFrame()
    sids = sorted(df.location.unique())
    df = df.set_index("location").sort_index()
    for sid in sids:
        df_pred = df.loc[sid]
        dates = df_pred[df_pred.output_type == "point"].target_end_date
        forecast_dates = df_pred[df_pred.output_type == "point"].reference_date
        preds = df_pred[df_pred.output_type == "point"].value
        targets = df_pred[df_pred.output_type == "point"].horizon

        if len(dates) == 0:
            mask = (df_pred.output_type == "quantile") & (df_pred["output_type_id"] == .025)
            dates = df_pred[mask].target_end_date
            forecast_dates = df_pred[mask].reference_date
            targets = df_pred[mask].horizon

        if len(preds) == 0:
            preds = df_pred[(df_pred.output_type == "quantile") & (df_pred["output_type_id"] == .5)].value

        ci025 = df_pred[(df_pred.output_type == "quantile") & (df_pred["output_type_id"] == .025)].value
        ci975 = df_pred[(df_pred.output_type == "quantile") & (df_pred["output_type_id"] == .975)].value
        ci05 = df_pred[(df_pred.output_type == "quantile") & (df_pred["output_type_id"] == .05)].value
        ci95 = df_pred[(df_pred.output_type == "quantile") & (df_pred["output_type_id"] == .95)].value

        if len(dates) == 0:
            continue

        preds = pd.DataFrame({
            "forecast_date": forecast_dates.to_numpy(),
            "model": [model_name] * len(dates),
            "location": [sid] * len(dates),
            "wks": [t for t in targets],
            "date": dates.to_numpy(),
            "value": preds.to_numpy(),
            "ci025": ci025.to_numpy(),
            "ci975": ci975.to_numpy(),
            "ci05": ci05.to_numpy(),
            "ci95": ci95.to_numpy(),
        })
        model_preds = pd.concat([model_preds, preds], axis=0)
    return model_preds


def process_truth(input_dir: Path):
    truth_path = input_dir / "target-data" / "target-hospital-admissions.csv"
    df = pd.read_csv(truth_path)
    df = df.set_index(["location", "date"], drop=True)
    return df


def process_data(input_dir: Path, output_dir: Path):
    fcast_dir = input_dir / "model-output"
    models = get_models(fcast_dir)

    all_datas = pd.DataFrame()
    for i, model in enumerate(models):
        if model.startswith("GT"):
            continue
        print(model)
        model_dir = fcast_dir / model
        datas = process_model_dir(model_dir)
        all_datas = pd.concat([all_datas, datas], axis=0)
    all_datas = all_datas.set_index(["location", "forecast_date", "wks"], drop=True)
    all_datas = all_datas.sort_index()

    gt_df = process_truth(input_dir)
    return gt_df, all_datas


def write_data(output_dir: Path, gt_df: pd.DataFrame, all_df: pd.DataFrame):
    idxs = gt_df.index.unique()
    locations = sorted({i for i, _ in idxs})
    dates = sorted({i for _, i in idxs})

    if not output_dir.exists():
        output_dir.mkdir()

    gt_df.reset_index().to_json(output_dir / "truth.json", orient="records")
    for location in locations:
        try:
            loc_df = all_df.loc[location].reset_index()
            loc_df.to_json(output_dir / "{}.json".format(location), orient="records")
            print("wrote location={}".format(location))
        except Exception as e:
            print("skipping location={}, {}".format(location, e))


def main(args):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        help="don't print status messages to stdout")
    parser.add_argument("--version", action="version",
                        version="%(prog)s " + __version__)
    parser.add_argument("--input_dir", type=Path, default=Path("FluSight-forecast-hub"),
                        help="directory containing clone of Flusight-forecast-data")
    parser.add_argument("--output_dir", type=Path, default=Path("output"),
                        help="directory where preprocessed results will be stored")
    options = parser.parse_args()

    gt_df, all_df = process_data(options.input_dir, options.output_dir)

    write_data(options.output_dir, gt_df, all_df)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
