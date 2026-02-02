"""Tests for Model.make() and Model.train() methods."""

import pytest


class TestModelMake:
    """Test Model.make() for training new models."""

    def test_make_dlc_model_basic(
        self,
        pv2_train,
        model,
        model_sel,
        model_params,
        skeleton,
        bodypart,
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test basic DLC model training via make()."""
        # This test requires a DLC project with labeled data
        # For now, we'll test the structure without actual training

        # Test that make() can be called
        # This will be implemented to actually train
        # For now, verify method exists
        assert hasattr(model, "make")

    def test_make_creates_nwb_file(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that make() creates an NWB file with model metadata."""
        # Will verify NWB creation after implementation

    def test_make_stores_model_path(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that make() stores the correct model_path."""
        # Will verify model_path is stored correctly

    def test_make_with_existing_training_dataset(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test make() when training dataset already exists."""
        # Should skip dataset creation and proceed to training

    def test_make_with_custom_training_params(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test make() with custom training parameters."""
        # Test passing maxiters, displayiters, saveiters, etc.


class TestModelTrain:
    """Test Model.train() for continued/additional training."""

    def test_train_creates_new_selection(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that train() creates new ModelSelection with parent_id."""
        # train() should:
        # 1. Create new ModelSelection
        # 2. Set parent_id to original model
        # 3. Trigger populate()

    def test_train_with_more_iterations(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test continuing training with additional iterations."""
        # Should allow training from existing snapshot

    def test_train_with_new_data(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test training with additional labeled frames."""
        # Should create new training dataset with combined data

    def test_train_parent_tracking(
        self,
        model,
        model_sel,
        skip_if_no_dlc,
    ):
        """Test that parent model is properly tracked."""
        # Verify parent_id is set correctly in ModelSelection

    def test_train_invalid_model(
        self,
        model,
    ):
        """Test error when training non-existent model."""
        with pytest.raises(ValueError, match="Model not found"):
            model.train({"model_id": "nonexistent"})


class TestTrainingDatasetManagement:
    """Test training dataset creation and management."""

    def test_create_training_dataset_new(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test creating a new training dataset."""
        # Should call DLC's create_training_dataset

    def test_create_training_dataset_exists(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test behavior when training dataset already exists."""
        # Should skip creation or append

    def test_training_dataset_with_augmentation(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test creating dataset with augmentation parameters."""
        # Should pass augmenter params to create_training_dataset


class TestModelMetadataStorage:
    """Test storing model metadata in NWB."""

    def test_model_metadata_in_nwb(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that model metadata is stored in NWB scratch space."""
        # Should store:
        # - Training params
        # - Training date
        # - Training duration
        # - Final loss
        # - Snapshot info

    def test_training_history_in_nwb(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that training history is stored."""
        # Should store loss curves, learning rate, etc.


class TestEndToEndTraining:
    """Test complete training workflows."""

    def test_e2e_train_new_model(
        self,
        pv2_train,
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test complete workflow: setup -> train -> evaluate."""
        # 1. Create ModelParams
        # 2. Create VidFileGroup with labeled videos
        # 3. Create ModelSelection
        # 4. Populate Model (triggers make())
        # 5. Verify Model entry created
        # 6. Verify NWB file exists

    def test_e2e_continue_training(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test complete workflow for continuing training."""
        # 1. Import or train initial model
        # 2. Call train() with additional iterations
        # 3. Verify new model created with parent_id
        # 4. Verify model improved

    def test_e2e_train_with_validation(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test training with validation split."""
        # Should use TrainingFraction from params


class TestTrainingMonitoring:
    """Test training progress monitoring."""

    def test_training_callback_logging(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that training progress is logged."""
        # Should use logger for training updates

    def test_training_early_stopping(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test early stopping based on validation loss."""
        # If supported by DLC

    def test_training_checkpoint_saving(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test that checkpoints are saved during training."""
        # Should save snapshots at specified intervals


class TestModelEvaluation:
    """Test Model.evaluate() functionality."""

    def test_evaluate_basic(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test basic model evaluation."""
        # Should call DLC evaluate_network
        # Should return dict with train/test errors
        assert hasattr(model, "evaluate")

    def test_evaluate_with_plotting(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test evaluation with labeled image generation."""
        # Should create labeled images in evaluation-results

    def test_evaluate_results_parsing(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test parsing of evaluation results CSV."""
        # Should parse train_error, test_error, p_cutoff, etc.

    def test_evaluate_invalid_model(
        self,
        model,
    ):
        """Test error when evaluating non-existent model."""
        with pytest.raises(ValueError, match="Model not found"):
            model.evaluate({"model_id": "nonexistent"})

    def test_evaluate_missing_dlc(
        self,
        model,
    ):
        """Test error when DLC not available."""
        # Should raise ImportError if evaluate_network not available


class TestTrainingHistory:
    """Test training history extraction and visualization."""

    def test_get_training_history(
        self,
        model,
        skip_if_no_dlc,
    ):
        """Test extracting training loss curves."""
        # Should read learning_stats.csv
        # Should return DataFrame with iteration, loss, learning_rate
        assert hasattr(model, "get_training_history")

    def test_get_training_history_missing(
        self,
        model,
    ):
        """Test behavior when training history not found."""
        # Should return None or empty DataFrame

    def test_plot_training_history(
        self,
        model,
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test plotting training loss curve."""
        # Should create matplotlib figure
        # Should save to file if path provided
        assert hasattr(model, "plot_training_history")

    def test_plot_training_history_save(
        self,
        model,
        skip_if_no_dlc,
        tmp_path,
    ):
        """Test saving training plot to file."""
        # Should create PNG file
        tmp_path / "training_plot.png"
        # Test that file is created
