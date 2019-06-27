from sklearn import metrics as mr

# 互信息(Mutual Information)
labels_true = [0, 0, 0, 1, 1, 1]
labels_pred = [0, 0, 1, 1, 2, 2]
MI = mr.adjusted_mutual_info_score(labels_true, labels_pred)  
print("Matual Information = "+str(MI))
# Matual Information = 0.2250422831983088