def f1_score(confusion_matrix):
    assert confusion_matrix.shape == (2, 2)
    true_positive = confusion_matrix[1, 1]
    false_positive = confusion_matrix[0, 1]
    false_negative = confusion_matrix[1, 0]
    return 2*true_positive/(2*true_positive+false_positive+false_negative)

if __name__ == '__main__':
    import torch
    import numpy as np
    from codesign.utils.plot_confusion_matrix import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    y_pred = torch.randint(0, 2, (10,))
    y_true = torch.randint(0, 2, (10,))
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    plot_confusion_matrix(
        cm=cm, 
        target_names=['healthy', 'unhealthy'], 
        title=f'F1 score = {f1_score(cm):.4f}, Accuracy = {100*(np.trace(cm)/np.sum(cm)):.4f}%',
        fname=f'./test_confusion_matrix.png',
        normalize=False
    )