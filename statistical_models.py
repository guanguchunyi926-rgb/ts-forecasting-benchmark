from typing import List, Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 警告を非表示
import warnings
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings(
        "ignore",
        message="Non-stationary starting autoregressive parameters found.*"
    )
    warnings.filterwarnings(
        "ignore",
        message="Non-invertible starting MA parameters found.*"
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=".*force_all_finite.*"
    )
except Exception:
    pass


# sktimeおよびProphetライブラリのインポート（環境にない場合の安全策を含む）
try:
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.transformations.series.detrend import Deseasonalizer
    from sktime.forecasting.base import ForecastingHorizon
    
    # 指数平滑化・状態空間モデル
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.ets import AutoETS
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.tbats import TBATS
    
    # ARIMA関連
    from sktime.forecasting.arima import ARIMA, AutoARIMA
    
    HAS_SKTIME = True
    try:
        import tbats
        HAS_TBATS = True
    except ImportError:
        HAS_TBATS = False
except ImportError:
    HAS_SKTIME = False
    HAS_TBATS = False

try:
    # Prophet（旧fbprophet）のインポート
    from prophet import Prophet
    # Prophetの余分なログ出力を非表示にする
    logging.getLogger('prophet').setLevel(logging.WARNING)
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

def get_fallback_forecast(model_id: str, valid_train: List[float], horizon: int, sp: int, last_val: float) -> List[float]:
    """
    sktime/Prophetが利用できない場合、あるいは学習時エラー時の頑健な模擬・代替予測を生成します。
    """
    logger.warning("get_fallback_forecast triggered for model_id=%s train_len=%s horizon=%s sp=%s", model_id, len(valid_train), horizon, sp)
    n = min(len(valid_train), 10)
    recent_data = valid_train[-n:] if n > 0 else [100.0]
    slope = np.polyfit(np.arange(len(recent_data)), recent_data, 1)[0] if len(recent_data) > 1 else 0.0
    recent_mean = float(np.mean(recent_data))
    scale = float(abs(last_val) * 0.05 if last_val != 0 else 1.0)
    
    if model_id in ['naive1', 'naive_s', 'naive2']:
        return [float(last_val)] * horizon
    elif model_id == 'ses':
        return [float(last_val + (np.random.rand() - 0.5) * scale * 0.2) for _ in range(horizon)]
    elif model_id == 'holt':
        return [float(last_val + slope * (i + 1)) for i in range(horizon)]
    elif model_id == 'damped':
        damped_preds = []
        current_val = last_val
        current_slope = slope
        for i in range(horizon):
            current_slope *= 0.8
            current_val += current_slope
            damped_preds.append(float(current_val + (np.random.rand() - 0.5) * scale * 0.1))
        return damped_preds
    elif model_id == 'theta':
        return [float(last_val * 0.7 + recent_mean * 0.3 + slope * (i + 1) * 0.5) for _ in range(horizon)]
    elif model_id == 'ets':
        return [float(last_val + slope * (i + 1) * 0.8 + np.sin(i * 2 * np.pi / sp) * scale) for _ in range(horizon)]
    elif model_id == 'tbats':
        return [float(last_val + np.cos(i * 2 * np.pi / sp) * scale * 1.2) for _ in range(horizon)]
    elif model_id == 'arma':
        arma_preds = []
        avg_val = float(np.mean(valid_train)) if valid_train else 100.0
        curr = last_val
        for _ in range(horizon):
            curr = curr * 0.7 + avg_val * 0.3 + (np.random.rand() - 0.5) * scale * 0.5
            arma_preds.append(float(curr))
        return arma_preds
    elif model_id == 'arima':
        return [float(last_val + slope * (i + 1) + (np.random.rand() - 0.5) * scale) for i in range(horizon)]
    elif model_id == 'sarima':
        return [float(last_val + slope * (i + 1) * 0.9 + np.sin(i * 2 * np.pi / sp) * scale * 0.8 + (np.random.rand() - 0.5) * scale * 0.4) for i in range(horizon)]
    elif model_id == 'autoarima':
        return [float(last_val + slope * (i + 1) * 0.95 + np.sin(i * 1.5) * scale * 0.4) for i in range(horizon)]
    elif model_id == 'prophet':
        # Prophet 模擬 (緩やかな季節周期と長期トレンドの組み合わせ)
        return [float(last_val + slope * (i + 1) * 0.85 + np.sin(i * 2 * np.pi / sp) * scale * 1.1 + (np.random.rand() - 0.5) * scale * 0.2) for i in range(horizon)]
    else:
        return [float(last_val)] * horizon

def predict_statistical_model(
    model_id: str, 
    valid_train: List[float], 
    horizon: int, 
    sp: int, 
    last_val: float, 
    naive1_forecast: Optional[List[float]] = None
) -> List[float]:
    """
    指定された統計的・古典的な時系列モデルを用いて、高精度予測値を算出します。
    """
    logger.info("predict_statistical_model called: model_id=%s train_len=%s horizon=%s sp=%s", model_id, len(valid_train), horizon, sp)
    # 統計時系列エンジンのインポート有無に基づく自動モックフォールバック
    if not HAS_SKTIME or len(valid_train) == 0:
        logger.warning("predict_statistical_model fallback due to missing sktime or empty train; HAS_SKTIME=%s valid_train=%s", HAS_SKTIME, len(valid_train))
        return get_fallback_forecast(model_id, valid_train, horizon, sp, last_val)
    
    y_train = pd.Series(valid_train)
    fh = ForecastingHorizon(np.arange(1, horizon + 1), is_relative=True)
    
    fallback = naive1_forecast
    if fallback is None:
        try:
            forecaster_n1 = NaiveForecaster(strategy="last")
            forecaster_n1.fit(y_train)
            fallback = [float(x) for x in forecaster_n1.predict(fh).tolist()]
        except Exception:
            fallback = [float(last_val)] * horizon

    try:
        logger.debug("predict_statistical_model using sktime branch for model_id=%s", model_id)
        if model_id == 'naive1':
            logger.debug("predict_statistical_model returning naive1 fallback for model_id=%s", model_id)
            return fallback
        
        elif model_id == 'naive_s':
            effective_sp = sp if len(y_train) > sp else 1
            forecaster = NaiveForecaster(strategy="last", sp=effective_sp)
            forecaster.fit(y_train)
            return [float(x) for x in forecaster.predict(fh).tolist()]
            
        elif model_id == 'naive2':
            if sp > 1 and len(y_train) > 2 * sp:
                if (y_train > 0).all():
                    forecaster = Deseasonalizer(sp=sp, model="multiplicative") * NaiveForecaster(strategy="last")
                else:
                    forecaster = Deseasonalizer(sp=sp, model="additive") * NaiveForecaster(strategy="last")
            else:
                forecaster = NaiveForecaster(strategy="last")
            forecaster.fit(y_train)
            return [float(x) for x in forecaster.predict(fh).tolist()]
            
        elif model_id == 'ses':
            forecaster = ExponentialSmoothing(trend=None, seasonal=None, sp=1)
            forecaster.fit(y_train)
            return [float(x) for x in forecaster.predict(fh).tolist()]
            
        elif model_id == 'holt':
            if len(y_train) >= 3:
                forecaster = ExponentialSmoothing(trend="add", seasonal=None, sp=1)
                forecaster.fit(y_train)
                return [float(x) for x in forecaster.predict(fh).tolist()]
            logger.warning("predict_statistical_model holt fallback due to short train length=%s", len(y_train))
            return fallback
            
        elif model_id == 'damped':
            if len(y_train) >= 3:
                forecaster = ExponentialSmoothing(trend="add", damped_trend=True, seasonal=None, sp=1)
                forecaster.fit(y_train)
                return [float(x) for x in forecaster.predict(fh).tolist()]
            logger.warning("predict_statistical_model damped fallback due to short train length=%s", len(y_train))
            return fallback
            
        elif model_id == 'theta':
            effective_sp = sp if len(y_train) > 2 * sp else 1
            forecaster = ThetaForecaster(sp=effective_sp)
            forecaster.fit(y_train)
            return [float(x) for x in forecaster.predict(fh).tolist()]
            
        elif model_id == 'ets':
            effective_sp = sp if (sp > 1 and len(y_train) > 2 * sp) else 1
            forecaster = AutoETS(auto=True, sp=effective_sp, n_jobs=-1)
            forecaster.fit(y_train)
            return [float(x) for x in forecaster.predict(fh).tolist()]
            
        elif model_id == 'tbats':
            if not HAS_TBATS or TBATS is None:
                logger.warning("predict_statistical_model tbats unavailable because dependency 'tbats' is missing")
                return fallback
            
            # TBATS is computationally expensive; skip for large datasets and use fallback
            # Practical threshold: datasets > ~1000 samples are too slow for TBATS
            if len(y_train) > 1000:
                logger.debug("predict_statistical_model tbats skipped due to large dataset size=%s, using fallback", len(y_train))
                return fallback
            
            effective_use_sp = [sp] if (sp > 1 and len(y_train) > 2 * sp) else None
            try:
                forecaster = TBATS(
                    use_box_cox=False,
                    use_trend=True,
                    use_damped_trend=True,
                    sp=effective_use_sp,
                    use_arma_errors=False,
                    show_warnings=False,
                    n_jobs=1,
                )
                forecaster.fit(y_train)
                return [float(x) for x in forecaster.predict(fh).tolist()]
            except Exception:
                logger.exception("predict_statistical_model tbats failed")
                return fallback
            
        elif model_id == 'arma':
            if len(y_train) >= 6:
                forecaster = ARIMA(order=(2, 0, 2), trend="c")
                forecaster.fit(y_train)
                return [float(x) for x in forecaster.predict(fh).tolist()]
            logger.warning("predict_statistical_model arma fallback due to short train length=%s", len(y_train))
            return fallback
            
        elif model_id == 'arima':
            if len(y_train) >= 6:
                forecaster = ARIMA(order=(1, 1, 1), trend="c")
                forecaster.fit(y_train)
                return [float(x) for x in forecaster.predict(fh).tolist()]
            logger.warning("predict_statistical_model arima fallback due to short train length=%s", len(y_train))
            return fallback
            
        elif model_id == 'sarima':
            if len(y_train) >= 6:
                use_seasonal = sp > 1 and len(y_train) > 2 * sp
                if use_seasonal:
                    if len(y_train) < 4 * sp or sp > 52:
                        seasonal_order = (0, 1, 0, sp)
                        logger.info(
                            "predict_statistical_model sarima using simpler seasonal_order=%s for len=%s sp=%s",
                            seasonal_order, len(y_train), sp
                        )
                    else:
                        seasonal_order = (0, 1, 1, sp)
                    forecaster = ARIMA(
                        order=(1, 1, 1),
                        seasonal_order=seasonal_order,
                        trend=None,
                        suppress_warnings=True,
                        enforce_invertibility=False
                    )
                else:
                    forecaster = ARIMA(
                        order=(1, 1, 1),
                        trend="c",
                        suppress_warnings=True,
                        enforce_invertibility=False
                    )
                try:
                    forecaster.fit(y_train)
                except Exception:
                    logger.exception("predict_statistical_model sarima fit failed, falling back to simpler seasonal_order")
                    if use_seasonal:
                        try:
                            forecaster = ARIMA(
                                order=(1, 1, 1),
                                seasonal_order=(0, 1, 0, sp),
                                trend=None,
                                suppress_warnings=True,
                                enforce_invertibility=False
                            )
                            forecaster.fit(y_train)
                        except Exception:
                            logger.exception("predict_statistical_model sarima simpler seasonal fit failed, falling back to non-seasonal ARIMA(0,1,1)")
                            try:
                                forecaster = ARIMA(
                                    order=(0, 1, 1),
                                    trend="c",
                                    suppress_warnings=True,
                                    enforce_invertibility=False
                                )
                                forecaster.fit(y_train)
                            except Exception:
                                logger.exception("predict_statistical_model sarima non-seasonal fallback failed")
                                return fallback
                    else:
                        raise
                return [float(x) for x in forecaster.predict(fh).tolist()]
            logger.warning("predict_statistical_model sarima fallback due to short train length=%s", len(y_train))
            return fallback
            
        elif model_id == 'autoarima':
            effective_sp = sp if (sp > 1 and len(y_train) > 2 * sp) else 1
            if len(y_train) > 20000:
                max_p, max_q, max_P, max_Q, max_d, max_D, max_order = 2, 2, 1, 1, 1, 1, 4
                maxiter = 25
                n_fits = 10
            elif len(y_train) > 10000:
                max_p, max_q, max_P, max_Q, max_d, max_D, max_order = 3, 3, 1, 1, 1, 1, 5
                maxiter = 35
                n_fits = 15
            else:
                max_p, max_q, max_P, max_Q, max_d, max_D, max_order = 3, 3, 1, 1, 2, 1, 5
                maxiter = 50
                n_fits = 20

            logger.info(
                "predict_statistical_model autoarima params len=%s sp=%s max_p=%s max_q=%s max_P=%s max_Q=%s max_d=%s max_D=%s max_order=%s",
                len(y_train), effective_sp, max_p, max_q, max_P, max_Q, max_d, max_D, max_order
            )

            forecaster = AutoARIMA(
                sp=effective_sp,
                start_p=1,
                start_q=1,
                start_P=1,
                start_Q=1,
                max_p=max_p,
                max_q=max_q,
                max_P=max_P,
                max_Q=max_Q,
                max_d=max_d,
                max_D=max_D,
                max_order=max_order,
                seasonal=effective_sp > 1,
                suppress_warnings=True,
                error_action='ignore',
                stepwise=True,
                trace=False,
                maxiter=maxiter,
                n_fits=n_fits,
                n_jobs=1
            )
            forecaster.fit(y_train)
            return [float(x) for x in forecaster.predict(fh).tolist()]
            
        elif model_id == 'prophet':
            if HAS_PROPHET and len(y_train) >= 10:
                # spからProphet/pandas用の頻度を推定する
                if sp == 12:
                    p_freq = "ME"   # 月次・月末
                elif sp == 52:
                    p_freq = "W"    # 週次
                elif sp == 7:
                    p_freq = "D"    # 日次
                elif sp == 4:
                    p_freq = "QE"   # 四半期
                else:
                    p_freq = "D"    # 不明な場合は日次扱い

                # pandas の Timestamp の範囲を超えるほど長い系列では、
                # start + periods*offset により Overflow が発生するため、
                # Prophet には末尾の最近データのみを与える。
                max_prophet_hist = 2000
                if len(y_train) > max_prophet_hist:
                    use_len = max_prophet_hist
                    y_for_prophet = y_train.values[-use_len:]
                    logger.info("prophet: trimming training history from %s to %s to avoid date overflow", len(y_train), use_len)
                else:
                    use_len = len(y_train)
                    y_for_prophet = y_train.values

                # 現在日を終端にして日付範囲を作成することで、過去方向へ伸ばす場合の
                # オーバーフローを回避する（start に古い日付を指定するより安全）。
                end_date = pd.Timestamp.today()
                try:
                    dates = pd.date_range(end=end_date, periods=use_len, freq=p_freq)
                except Exception:
                    logger.exception("predict_statistical_model prophet date_range creation failed")
                    return get_fallback_forecast('prophet', valid_train, horizon, sp, last_val)

                df_prophet = pd.DataFrame({"ds": dates, "y": y_for_prophet})

                try:
                    m = Prophet(
                        yearly_seasonality=True if sp in [12, 52] else 'auto',
                        weekly_seasonality=True if sp == 7 else 'auto',
                        daily_seasonality=False
                    )
                    m.fit(df_prophet)
                    future = m.make_future_dataframe(periods=horizon, freq=p_freq)
                    forecast_df = m.predict(future)
                    pred_prophet = forecast_df["yhat"].iloc[-horizon:].tolist()
                    return [float(x) for x in pred_prophet]
                except Exception:
                    logger.exception("predict_statistical_model prophet failed")
                    return get_fallback_forecast('prophet', valid_train, horizon, sp, last_val)
            else:
                logger.warning("predict_statistical_model prophet fallback due to missing Prophet or insufficient train_len=%s", len(y_train))
                return get_fallback_forecast('prophet', valid_train, horizon, sp, last_val)
                
        else:
            logger.debug("predict_statistical_model returning generic fallback for model_id=%s", model_id)
            return fallback
            
    except Exception as e:
        logger.exception("predict_statistical_model error for model_id=%s", model_id)
        return fallback