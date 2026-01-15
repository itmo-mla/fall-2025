import numpy as np
from SVM import SVM   
from utils import load_base,visualize_t_sne_3d,visualize_t_sne_and_pca
from sklearn.svm import SVC

#изначально точность отличалась лишь потому что svm от sklearn автоматически масштабирует данные
X_train, X_test, y_train, y_test = load_base()

 
svm_custom = SVM(c=10,kernel_type='linear')
svm_custom.fit(X_train, y_train)
y_pred_custom = svm_custom.predict(X_test)
accuracy_custom = np.mean(y_pred_custom == y_test)
print("Custom SVM accuracy:", accuracy_custom)

 
svm_sklearn = SVC(C=10, kernel='linear')  
svm_sklearn.fit(X_train, y_train)
y_pred_sklearn = svm_sklearn.predict(X_test)
accuracy_sklearn = np.mean(y_pred_sklearn == y_test)
print("Sklearn SVM accuracy:", accuracy_sklearn)
  
visualize_t_sne_3d(X_test,y_pred_custom   )
visualize_t_sne_and_pca(X_test,y_pred_custom,svm_custom.x_ref)