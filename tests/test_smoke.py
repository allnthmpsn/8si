from training import train_model1


def test_trainer_runs_end_to_end(tmp_path):
    acc = train_model1.main(
        train_start='2019-01-01',
        train_cutoff='2024-06-30',
        output_dir=str(tmp_path),
    )

    assert 0.4 < acc < 0.9

    for fname in [
        'ufc_model_best.pkl',
        'ufc_model_xgb.pkl',
        'feature_columns_best.pkl',
        'model_metadata.json',
        'elo_ratings_history.csv',
        'elo_current.csv',
    ]:
        assert (tmp_path / fname).exists(), f'missing artifact: {fname}'
