from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(X_train, y_train, pca_components=150):
    scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                    whiten=True, random_state=42)),
    ])

    X_train_scaled = scaler.fit_transform(X_train)

    # เพิ่ม probability=True เพื่อให้สามารถคำนวณ % ความมั่นใจได้
    model = SVC(
        kernel="rbf", C=10, gamma="scale", cache_size=1000, probability=True
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_svm(model, scaler, X_test):

    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)

    return predictions