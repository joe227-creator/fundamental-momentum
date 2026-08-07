"""Run one fixed-score Optuna direction."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import numpy as np
import optuna
from research_score import _evaluate_result, _load_reference, _prepare_strategy, _select, _write_artifacts

def _evaluate(config_path, base_params, overrides, baseline_turnover):
    params=dict(base_params); params.update(overrides)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(params, handle); path=handle.name
    try:
        prepared=_prepare_strategy(config_path,path); metrics,_,_,_=_evaluate_result(prepared,_select(prepared),baseline_turnover); return metrics,prepared
    finally: Path(path).unlink(missing_ok=True)

def run_optuna(config_path, params_path, reference_path, artifact_dir, optuna_config_path, perturb_runs, perturb_sigma, seed):
    direction=json.loads(Path(optuna_config_path).read_text(encoding="utf-8")); base=json.loads(Path(params_path).read_text(encoding="utf-8")); ref=_load_reference(Path(reference_path)); root=Path(artifact_dir); root.mkdir(parents=True,exist_ok=True)
    study=optuna.create_study(study_name=direction["study_name"],storage=f"sqlite:///{root/'optuna_confidence_conditioned_breadth.db'}",load_if_exists=False,direction="maximize",sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.MedianPruner(n_startup_trials=2))
    def objective(trial):
        value=trial.suggest_float("breadth_threshold",0.0,0.5); metrics,_=_evaluate(config_path,base,{"breadth_threshold":value},ref); trial.report(metrics["research_score"],step=0)
        if trial.should_prune(): raise optuna.TrialPruned()
        return metrics["research_score"]
    study.optimize(objective,n_trials=6,gc_after_trial=True); best=dict(study.best_params); bm,prepared=_evaluate(config_path,base,best,ref); seeds=[]
    for i in range(max(0,int(perturb_runs))):
        m,_,_,_=_evaluate_result(prepared,_select(prepared,seed=seed+1000+i,sigma=perturb_sigma),ref); m["seed"]=seed+1000+i; seeds.append(m)
    scores=np.array([m["research_score"] for m in seeds]); bm.update({"perturb_score_mean":float(scores.mean()),"perturb_score_std":float(scores.std(ddof=0)),"perturb_score_min":float(scores.min()),"perturb_score_max":float(scores.max())}); _,curve,windows,trades=_evaluate_result(prepared,_select(prepared),ref); (root/"optuna_best_params.json").write_text(json.dumps(best,indent=2),encoding="utf-8"); study.trials_dataframe().to_csv(root/"optuna_trials.csv",index=False); _write_artifacts(root,bm,curve,windows,trades,seeds,prepared["cfg"])
    for key in ("research_score","mean_rolling_6m_return","return_on_risk","win_rate","maximum_drawdown","sharpe","turnover","perturb_score_mean","perturb_score_std","perturb_score_min","perturb_score_max"): print(f"METRIC {key}={bm[key]}")
    print(f"ARTIFACT_DIR {artifact_dir}"); return 0
