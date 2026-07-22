from Drift.psi import compute_all_feature_psi,compute_all_feature_ks
from performance.metrics import compute_performance_report
from root_cause.correlation import generate_root_cause_report
from root_cause.correlation import generate_root_cause_report, get_time_series, get_drift_timeseries, get_performance_timeseries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/performance/timeseries")
def get_performance_timeseries_endpoint(baseline_start: str, baseline_end: str):
    day_starts, day_ends = get_time_series(0, 9)
    accuracy_series = get_performance_timeseries(day_starts, day_ends)

    return [
        {"day": i, "start": day_starts[i], "accuracy": accuracy_series[i]}
        for i in range(len(day_starts))
    ]

@app.get("/drift/timeseries")
def get_drift_timeseries_endpoint(baseline_start: str, baseline_end: str):
    day_starts, day_ends = get_time_series(0, 9)
    drift_series = get_drift_timeseries(baseline_start, baseline_end, day_starts, day_ends)

    return {
        "days": day_starts,
        "features": drift_series
    }


@app.get("/drift")
def get_drift(baseline_start: str, baseline_end: str, comparison_start: str, comparison_end: str):
    result = {}
    result["psi"] = compute_all_feature_psi(baseline_start, baseline_end, comparison_start, comparison_end)
    result["Ks"] = compute_all_feature_ks(baseline_start, baseline_end, comparison_start, comparison_end)
    
    return result

@app.get("/performance")
def get_performance(baseline_start: str, baseline_end: str, comparison_start: str, comparison_end: str):
    return compute_performance_report(baseline_start, baseline_end, comparison_start, comparison_end)

@app.get("/rootcause")
def get_rootcause(baseline_start: str, baseline_end: str):
    report = generate_root_cause_report(baseline_start, baseline_end)
    formatted = [{"feature": feature, **stats} for feature, stats in report]
    return formatted