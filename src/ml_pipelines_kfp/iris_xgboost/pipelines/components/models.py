from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

@component(base_image="python:3.10", 
    packages_to_install=[
        "pandas==2.0.0",
        "scikit-learn==1.5.1",
        "numpy==1.23.0",
        "joblib==1.4.2",
    ],
)
def decision_tree(
    train_dataset: Input[Dataset],
    metrics: Output[Metrics],
    output_model: Output[Model],
):
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    train = pd.read_csv(train_dataset.path)

    X_train, X_test, y_train, y_test = train_test_split(
        train.drop("Species", axis=1),
        train["Species"],
        test_size=0.2,
        random_state=42,
    )

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    metrics.log_metric("accuracy", (acc))

    joblib.dump(model, output_model.path)


@component(base_image="python:3.10", 
    packages_to_install=[
        "pandas==2.0.0",
        "scikit-learn==1.5.1",
        "numpy==1.23.0",
        "joblib==1.4.2",
    ],
)
def random_forest(
    train_dataset: Input[Dataset],
    metrics: Output[Metrics],
    output_model: Output[Model],
):
    import joblib
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    train = pd.read_csv(train_dataset.path)

    X_train, X_test, y_train, y_test = train_test_split(
        train.drop("Species", axis=1),
        train["Species"],
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    metrics.log_metric("accuracy", (acc))

    joblib.dump(model, output_model.path)