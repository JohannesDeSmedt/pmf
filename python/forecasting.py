import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg as AR
from arch.univariate import arch_model as ARCH
from statsmodels.tsa.holtwinters import SimpleExpSmoothing as SES, ExponentialSmoothing as HW
from sklearn.preprocessing import MinMaxScaler


class Scaler:


    def __init__(self, low, up, array_in):
        X = array_in.copy()

        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        self.low = low
        self.up = up


    def scale(self, X_in):
        X = X_in.copy()
        X_std = (X - self.x_min) / (self.x_max - self.x_min)
        X_scaled = X_std * (self.up - self.low) + self.low
        return X_scaled

    def reverse(self, X_in):
        X = X_in.copy()
        X = X + self.low
        X = X * (self.up - self.low)
        X_ret =  (self.x_max - self.x_min) / (X - self.x_min)
        
        return X_ret


class NAVf:

    class Fitted:

        def __init__(self, x):
            self.x = x

        def forecast(self, horizon):
            mean = np.mean(self.x)
            # print('Naive:', np.full(shape=(1, horizon), fill_value=mean))
            return np.full(shape=(1, horizon), fill_value=mean)

    def __init__(self):
        pass

    def fit(self, x, horizon):
        return self.Fitted(x).forecast(horizon)[0]


class ARf:

    def __init__(self, d):
        self.d = d

    def fit(self, x, horizon):
        fit = AR(x, lags=self.d, old_names=False).fit()
        point_forecast = fit.get_prediction(len(x), len(x)+horizon-1)

        # print(f'AR {self.d}:', point_forecast.predicted_mean)
        # print(point_forecast.conf_int(0.05))

        return point_forecast.predicted_mean


class ARIMAf:

    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q

    def fit(self, x, horizon):
        fit = ARIMA(x, order=(self.p, self.d, self.q)).fit()
        point_forecast = fit.get_forecast(horizon)
        conf_int = fit.conf_int()

        # print('ARIMA:', point_forecast.predicted_mean)
        # print(point_forecast.conf_int())
        return point_forecast.predicted_mean


class ESf:

    def __init__(self):
        pass

    def fit(self, x, horizon):
        fit = ExponentialSmoothing(x).fit()
        point_forecast = fit.get_prediction(len(x), len(x)+horizon-1)
        print('ETS', point_forecast.predicted_mean)
        # print('ETS', point_forecast.conf_int())

        return point_forecast.predicted_mean


class SESf:

    def __init__(self):
        pass

    def fit(self, x, horizon):
        fit = SES(x, initialization_method='estimated').fit()
        # print('SES', fit.forecast(horizon))

        return fit.forecast(horizon)


class HWf:

    def __init__(self):
        pass

    def fit(self, x, horizon):
        fit = HW(x, initialization_method='estimated').fit()

        # print('HW', fit.forecast(horizon))
        return fit.forecast(horizon)


class GARCHf:

    def __init__(self):
        pass

    def fit(self, x, horizon):
        garch = ARCH(x)
        model = garch.fit(disp='off')
        y_hat = model.forecast(horizon=horizon)

        pred = y_hat.mean.iloc[[-1]].values[0]
        var = np.sqrt(y_hat.variance.iloc[[-1]].values) * 1.96
        conf_low = pred + var[0]
        conf_high = pred - var[0]
        return pred
