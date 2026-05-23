$QrpPrivateRunRoots = @(
    "tensor_long_only=reports/private_real/factor_evaluation/smoke_2023_2025_tensor_long_only/evaluation_runs",
    "tensor_long_short=reports/private_real/factor_evaluation/smoke_2023_2025_tensor_long_short/evaluation_runs",
    "dataframe_long_only=reports/private_real/factor_evaluation/smoke_2023_2025_dataframe_long_only/evaluation_runs",
    "dataframe_long_short=reports/private_real/factor_evaluation/smoke_2023_2025_dataframe_long_short/evaluation_runs"
) -join [System.IO.Path]::PathSeparator
