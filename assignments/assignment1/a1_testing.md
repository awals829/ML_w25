## Assignment 1 test data
If you are interested in seeing how your model performed on the held out test data and reproducing the figures I showed in class, download the file `data/housing_holdout.csv` and paste the following into your `if __name__ == "__main__":` block of your `prod.py` file, then run it with `python prod.py`:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
data = pd.read_csv("../housing_holdout.csv")
y = data.iloc[:, 1].values
X = data.drop(columns="ASSESSED_VALUE")
y_est = predict(X)

ax = plt.subplot(1,1,1)
ax.scatter(y, y_pred, s=1, alpha=0.3)
ax.set_xlim([0, 2e6])
ax.set_ylim([0, 2e6])
ax.set_xlabel("Actual Values ($)")
ax.set_ylabel("Predicted Values ($)")
ax.plot([0, 2e6], [0, 2e6], 'k--', lw=1)
ax.set_title(f"MAE = ${mae(y, y_pred):,.2f}")
plt.show()
```

If you get the message `FigureCanvasAgg is non-interactive and thus cannot be shown`, you may need to `pip install pyqt6`.