def compute_shap_explanations(pipeline, X_sample, nsamples=100):
    """
    Opcional: calcula valores SHAP para explicar predicciones.
    Requiere shap paquete instalado: pip install shap
    Devuelve objecto explainer + shap_values.
    """
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap not installed. Install with `pip install shap` to use explanations.") from e

    # extract classifier from pipeline
    clf = pipeline.named_steps["clf"]
    # get preprocessed matrix
    preproc = pipeline.named_steps["preproc"]
    X_proc = preproc.transform(X_sample)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_proc, nsamples=nsamples)
    return explainer, shap_values
