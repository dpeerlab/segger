"""Optuna objective function for Segger hyperparameter optimization.

This module implements the objective function that:
1. Samples hyperparameters from the search space
2. Trains a Segger model with the sampled config
3. Evaluates using segbench metrics
4. Returns composite objectives for optimization
"""

from pathlib import Path
from typing import Any, Literal
import optuna

from .search_space import SEARCH_SPACE, SEARCH_SPACE_QUICK, sample_hyperparameters
from .metrics import compute_fast_metrics, compute_composites, scalarize_composites
from .samplers import get_sampler, get_pruner, get_study_directions


class SeggerObjective:
    """Optuna objective function for Segger hyperparameter optimization.

    This class caches heavy imports to avoid repeated import overhead.
    """

    # Class-level import cache
    _imports_cached = False
    _Trainer = None
    _CSVLogger = None
    _ModelCheckpoint = None
    _EarlyStopping = None
    _ISTDataModule = None
    _LitISTEncoder = None
    _ISTSegmentationWriter = None
    _PyTorchLightningPruningCallback = None

    @classmethod
    def _ensure_imports(cls):
        """Cache heavy imports at class level for reuse across trials."""
        if cls._imports_cached:
            return

        import warnings
        warnings.filterwarnings("ignore")

        from lightning.pytorch.plugins.environments import SLURMEnvironment
        SLURMEnvironment.detect = lambda: False

        from lightning.pytorch import Trainer
        from lightning.pytorch.loggers import CSVLogger
        from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

        from ..data import ISTDataModule
        from ..models import LitISTEncoder
        from ..data import ISTSegmentationWriter

        cls._Trainer = Trainer
        cls._CSVLogger = CSVLogger
        cls._ModelCheckpoint = ModelCheckpoint
        cls._EarlyStopping = EarlyStopping
        cls._ISTDataModule = ISTDataModule
        cls._LitISTEncoder = LitISTEncoder
        cls._ISTSegmentationWriter = ISTSegmentationWriter

        # Optional: Optuna Lightning pruning callback
        try:
            from optuna.integration import PyTorchLightningPruningCallback
            cls._PyTorchLightningPruningCallback = PyTorchLightningPruningCallback
        except ImportError:
            cls._PyTorchLightningPruningCallback = None

        cls._imports_cached = True
    """Optuna objective function for Segger hyperparameter optimization.

    Parameters
    ----------
    input_directory : Path
        Directory containing input data.
    output_directory : Path
        Directory for output files.
    n_epochs : int
        Number of training epochs per trial.
    search_space : dict, optional
        Custom search space. If None, uses SEARCH_SPACE.
    n_objectives : int
        Number of objectives (1 for single, 5 for multi-objective).
    fast_metrics : bool
        If True, use fast metric subset for evaluation.
    reference_path : Path, optional
        Path to scRNA-seq reference for metrics.
    precision : str
        Training precision ('32', '16-mixed', 'bf16-mixed').
    fidelity : float
        Data fraction for multi-fidelity HPO (0.0-1.0). Default 1.0 uses all data.
    early_stopping_patience : int
        Number of epochs with no improvement before stopping. 0 disables early stopping.
    """

    def __init__(
        self,
        input_directory: Path,
        output_directory: Path,
        n_epochs: int = 5,
        search_space: dict[str, dict[str, Any]] | None = None,
        n_objectives: int = 1,
        fast_metrics: bool = True,
        reference_path: Path | None = None,
        precision: str = "16-mixed",
        fidelity: float = 1.0,
        early_stopping_patience: int = 3,
    ):
        self.input_directory = Path(input_directory)
        self.output_directory = Path(output_directory)
        self.n_epochs = n_epochs
        self.search_space = search_space or SEARCH_SPACE
        self.n_objectives = n_objectives
        self.fast_metrics = fast_metrics
        self.reference_path = reference_path
        self.precision = precision
        self.fidelity = fidelity
        self.early_stopping_patience = early_stopping_patience

        # Ensure imports are cached
        self._ensure_imports()

    def __call__(self, trial: optuna.Trial) -> float | tuple[float, ...]:
        """Execute a single HPO trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.

        Returns
        -------
        float or tuple
            Single objective value or tuple of 5 composite objectives.
        """
        # Sample hyperparameters
        params = sample_hyperparameters(trial, self.search_space)

        # Create trial-specific output directory
        trial_dir = self.output_directory / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Train model
            self._train_model(params, trial_dir, trial)

            # Compute metrics
            predictions_path = trial_dir / "predictions.parquet"
            if not predictions_path.exists():
                # Training failed, return worst score
                return self._worst_score()

            metrics = compute_fast_metrics(
                predictions_path=predictions_path,
                transcripts_path=self.input_directory / "transcripts.parquet",
                reference_path=self.reference_path,
            )

            # Compute composites
            composites = compute_composites(metrics)

            if self.n_objectives == 1:
                return scalarize_composites(composites)
            else:
                return composites

        except Exception:
            # Return worst score for failed trials
            return self._worst_score()

    def _train_model(
        self,
        params: dict[str, Any],
        output_dir: Path,
        trial: optuna.Trial,
    ) -> None:
        """Train a Segger model with the given parameters.

        Parameters
        ----------
        params : dict
            Hyperparameters to use.
        output_dir : Path
            Output directory for this trial.
        trial : optuna.Trial
            Optuna trial for pruning callbacks.
        """
        # Use cached imports
        Trainer = self._Trainer
        CSVLogger = self._CSVLogger
        EarlyStopping = self._EarlyStopping
        ISTDataModule = self._ISTDataModule
        LitISTEncoder = self._LitISTEncoder
        ISTSegmentationWriter = self._ISTSegmentationWriter
        PyTorchLightningPruningCallback = self._PyTorchLightningPruningCallback

        # Setup data module with sampled params
        datamodule = ISTDataModule(
            input_directory=self.input_directory,
            num_workers=0,  # Reduce overhead for HPO
            transcripts_graph_max_k=params.get("transcripts_max_k", 5),
            transcripts_graph_max_dist=params.get("transcripts_max_dist", 5.0),
            prediction_graph_scale_factor=params.get("prediction_scale_factor", 2.0),
        )

        # Apply fidelity (data subsampling) if < 1.0
        if self.fidelity < 1.0 and hasattr(datamodule, 'subsample'):
            datamodule.subsample(fraction=self.fidelity)

        # Setup model with sampled params
        n_genes = datamodule.ad.shape[1]
        model = LitISTEncoder(
            n_genes=n_genes,
            n_mid_layers=params.get("n_mid_layers", 2),
            n_heads=params.get("n_heads", 2),
            in_channels=params.get("node_representation_dim", 64),
            hidden_channels=params.get("hidden_channels", 64),
            out_channels=params.get("out_channels", 64),
            learning_rate=params.get("learning_rate", 1e-3),
            lr_scheduler=params.get("lr_scheduler", "none"),
            sg_loss_type=params.get("segmentation_loss", "triplet"),
            tx_margin=params.get("transcripts_margin", 0.3),
            sg_margin=params.get("segmentation_margin", 0.4),
            tx_weight_end=params.get("transcripts_loss_weight_end", 1.0),
            bd_weight_end=params.get("cells_loss_weight_end", 1.0),
            sg_weight_end=params.get("segmentation_loss_weight_end", 0.5),
        )

        # Setup callbacks
        callbacks = []
        writer = ISTSegmentationWriter(output_dir)
        callbacks.append(writer)

        # Add EarlyStopping callback
        if self.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor="val:loss_sg",
                patience=self.early_stopping_patience,
                mode="min",
                verbose=False,
            )
            callbacks.append(early_stopping)

        # Add Optuna pruning callback if available
        if PyTorchLightningPruningCallback is not None:
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val:loss_sg")
            callbacks.append(pruning_callback)

        # Setup trainer
        trainer = Trainer(
            logger=CSVLogger(output_dir),
            max_epochs=self.n_epochs,
            callbacks=callbacks,
            precision=self.precision,
            enable_progress_bar=False,
            enable_model_summary=False,
            reload_dataloaders_every_n_epochs=1,
        )

        # Train
        trainer.fit(model=model, datamodule=datamodule)

        # Predict
        trainer.predict(model=model, datamodule=datamodule)

        # Write predictions
        writer.write_on_epoch_end(
            trainer=trainer,
            pl_module=model,
            predictions=[],
            batch_indices=[],
        )

    def _worst_score(self) -> float | tuple[float, ...]:
        """Return worst possible score for failed trials."""
        if self.n_objectives == 1:
            return 0.0
        else:
            return (0.0,) * 5


def create_study(
    study_name: str = "segger_hpo",
    storage: str | None = None,
    sampler_type: Literal["tpe", "nsga3", "random", "cmaes"] = "tpe",
    pruner_type: Literal["median", "hyperband", "none"] = "median",
    n_objectives: int = 1,
    seed: int | None = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    """Create or load an Optuna study for Segger HPO.

    Parameters
    ----------
    study_name : str
        Name of the study.
    storage : str, optional
        Database URL for study persistence (e.g., "sqlite:///hpo.db").
    sampler_type : str
        Type of sampler to use.
    pruner_type : str
        Type of pruner to use.
    n_objectives : int
        Number of objectives.
    seed : int, optional
        Random seed for reproducibility.
    load_if_exists : bool
        If True, load existing study with the same name.

    Returns
    -------
    optuna.Study
        Configured Optuna study.
    """
    sampler = get_sampler(sampler_type, n_objectives, seed)
    pruner = get_pruner(pruner_type)
    directions = get_study_directions(n_objectives)

    if n_objectives == 1:
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction=directions[0],
            load_if_exists=load_if_exists,
        )
    else:
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            directions=directions,
            load_if_exists=load_if_exists,
        )


def run_hpo(
    input_directory: Path,
    output_directory: Path,
    n_trials: int = 100,
    n_epochs: int = 5,
    n_jobs: int = 1,
    n_objectives: int = 1,
    sampler_type: Literal["tpe", "nsga3", "random", "cmaes"] = "tpe",
    pruner_type: Literal["hyperband", "median", "none"] = "hyperband",
    search_space: dict[str, dict[str, Any]] | None = None,
    quick_search: bool = False,
    reference_path: Path | None = None,
    precision: str = "16-mixed",
    study_name: str = "segger_hpo",
    storage: str | None = None,
    seed: int | None = None,
    show_progress_bar: bool = True,
    fidelity: float = 1.0,
    early_stopping_patience: int = 3,
    workflow: Literal["none", "smart"] = "none",
) -> optuna.Study:
    """Run hyperparameter optimization for Segger.

    Parameters
    ----------
    input_directory : Path
        Directory containing input data.
    output_directory : Path
        Directory for output files.
    n_trials : int
        Number of trials to run.
    n_epochs : int
        Number of training epochs per trial.
    n_jobs : int
        Number of parallel trials.
    n_objectives : int
        Number of objectives (1 for single, 5 for multi-objective).
    sampler_type : str
        Type of sampler to use.
    pruner_type : str
        Type of pruner to use. Default is 'hyperband' (best for TPE).
    search_space : dict, optional
        Custom search space.
    quick_search : bool
        If True, use reduced search space for faster HPO.
    reference_path : Path, optional
        Path to scRNA-seq reference for metrics.
    precision : str
        Training precision.
    study_name : str
        Name of the study.
    storage : str, optional
        Database URL for study persistence.
    seed : int, optional
        Random seed for reproducibility.
    show_progress_bar : bool
        If True, show progress bar during optimization.
    fidelity : float
        Data fraction for multi-fidelity HPO (0.0-1.0). Default 1.0 uses all data.
    early_stopping_patience : int
        Epochs with no improvement before stopping. 0 disables.
    workflow : str
        HPO workflow: 'none' for standard, 'smart' for two-stage multi-fidelity.

    Returns
    -------
    optuna.Study
        Completed Optuna study with results.
    """
    # Handle smart workflow (two-stage multi-fidelity)
    if workflow == "smart":
        return _run_smart_workflow(
            input_directory=input_directory,
            output_directory=output_directory,
            n_trials=n_trials,
            n_epochs=n_epochs,
            n_jobs=n_jobs,
            n_objectives=n_objectives,
            sampler_type=sampler_type,
            pruner_type=pruner_type,
            search_space=search_space,
            quick_search=quick_search,
            reference_path=reference_path,
            precision=precision,
            study_name=study_name,
            storage=storage,
            seed=seed,
            show_progress_bar=show_progress_bar,
            early_stopping_patience=early_stopping_patience,
        )

    # Use quick search space if requested
    if quick_search and search_space is None:
        search_space = SEARCH_SPACE_QUICK

    # Create output directory
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Create study
    study = create_study(
        study_name=study_name,
        storage=storage,
        sampler_type=sampler_type,
        pruner_type=pruner_type,
        n_objectives=n_objectives,
        seed=seed,
    )

    # Create objective
    objective = SeggerObjective(
        input_directory=input_directory,
        output_directory=output_directory,
        n_epochs=n_epochs,
        search_space=search_space,
        n_objectives=n_objectives,
        reference_path=reference_path,
        precision=precision,
        fidelity=fidelity,
        early_stopping_patience=early_stopping_patience,
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # Save results
    _save_study_results(study, output_directory, n_objectives)

    return study


def _run_smart_workflow(
    input_directory: Path,
    output_directory: Path,
    n_trials: int,
    n_epochs: int,
    n_jobs: int,
    n_objectives: int,
    sampler_type: str,
    pruner_type: str,
    search_space: dict | None,
    quick_search: bool,
    reference_path: Path | None,
    precision: str,
    study_name: str,
    storage: str | None,
    seed: int | None,
    show_progress_bar: bool,
    early_stopping_patience: int,
) -> optuna.Study:
    """Run two-stage smart HPO workflow.

    Stage 1: Fast exploration on 20% data with 3-5 epochs
    Stage 2: Refinement of top 10 configs on full data
    """
    import json

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Stage 1: Exploration (20% data, 3 epochs)
    stage1_dir = output_directory / "stage1_exploration"
    stage1_epochs = min(n_epochs, 3)

    print(f"Stage 1: Exploring {n_trials} trials on 20% data ({stage1_epochs} epochs)...")
    stage1_study = run_hpo(
        input_directory=input_directory,
        output_directory=stage1_dir,
        n_trials=n_trials,
        n_epochs=stage1_epochs,
        n_jobs=n_jobs,
        n_objectives=n_objectives,
        sampler_type=sampler_type,
        pruner_type=pruner_type,
        search_space=search_space,
        quick_search=quick_search,
        reference_path=reference_path,
        precision=precision,
        study_name=f"{study_name}_stage1",
        storage=storage,
        seed=seed,
        show_progress_bar=show_progress_bar,
        fidelity=0.2,  # 20% of data
        early_stopping_patience=early_stopping_patience,
        workflow="none",  # Don't recurse
    )

    # Get top 10 configurations from stage 1
    if n_objectives == 1:
        top_trials = sorted(stage1_study.trials, key=lambda t: t.value or 0, reverse=True)[:10]
    else:
        top_trials = stage1_study.best_trials[:10]

    top_params = [t.params for t in top_trials]

    # Save stage 1 results
    with open(output_directory / "stage1_top_params.json", "w") as f:
        json.dump(top_params, f, indent=2, default=str)

    # Stage 2: Refinement (full data, full epochs)
    stage2_dir = output_directory / "stage2_refinement"
    print(f"\nStage 2: Refining top {len(top_params)} configs on full data ({n_epochs} epochs)...")

    stage2_study = create_study(
        study_name=f"{study_name}_stage2",
        storage=storage,
        sampler_type=sampler_type,
        pruner_type="none",  # Don't prune in refinement
        n_objectives=n_objectives,
        seed=seed,
    )

    # Enqueue top configurations from stage 1
    for params in top_params:
        stage2_study.enqueue_trial(params)

    objective = SeggerObjective(
        input_directory=input_directory,
        output_directory=stage2_dir,
        n_epochs=n_epochs,
        search_space=search_space,
        n_objectives=n_objectives,
        reference_path=reference_path,
        precision=precision,
        fidelity=1.0,  # Full data
        early_stopping_patience=0,  # No early stopping in refinement
    )

    stage2_study.optimize(
        objective,
        n_trials=len(top_params),
        n_jobs=n_jobs,
        show_progress_bar=show_progress_bar,
    )

    # Save final results
    _save_study_results(stage2_study, output_directory, n_objectives)

    return stage2_study


def _save_study_results(
    study: optuna.Study,
    output_directory: Path,
    n_objectives: int,
) -> None:
    """Save study results to files.

    Parameters
    ----------
    study : optuna.Study
        Completed Optuna study.
    output_directory : Path
        Output directory.
    n_objectives : int
        Number of objectives.
    """
    import json

    output_directory = Path(output_directory)

    # Save trials as CSV
    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_directory / "hpo_trials.csv", index=False)
    except Exception:
        pass

    # Save best parameters
    if n_objectives == 1:
        best_params = study.best_params
        best_value = study.best_value
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": len(study.trials),
        }
    else:
        # Multi-objective: get Pareto front
        pareto_trials = study.best_trials
        results = {
            "pareto_front": [
                {
                    "params": t.params,
                    "values": t.values,
                }
                for t in pareto_trials
            ],
            "n_trials": len(study.trials),
            "n_pareto": len(pareto_trials),
        }

    with open(output_directory / "hpo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
