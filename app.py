import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from datetime import timedelta
import copy
from pyswarms.single.global_best import GlobalBestPSO

# Set Seed
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

st.set_page_config(layout="wide")

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.title("Stock Settings")

ticker_input = st.sidebar.text_input(
    "Masukkan ticker saham (contoh: BBCA)",
    "SIDO"
)

# otomatis tambah .JK jika belum ada
if ".JK" not in ticker_input:
    ticker = ticker_input.upper() + ".JK"
else:
    ticker = ticker_input.upper()

today = datetime.date.today()

start_date = st.sidebar.date_input(
    "Start Date",
    datetime.date(2019,7,1)
)

end_date = st.sidebar.date_input(
    "End Date",
    datetime.date(2025,7,1)
)

section = st.sidebar.radio(
    "Select Section",
    ["Informasi Data", "In-Depth Analysis", "Hasil Forecast"]
)

# =============================
# LOAD DATA
# =============================
@st.cache_resource(ttl=3600)
def load_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False
    )

    if df.empty:
        return pd.DataFrame()

    # Jika MultiIndex kolom
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ambil hanya Close
    df = df[['Close']].copy()

    df.dropna(inplace=True)

    return df

# Load data
data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("Data tidak ditemukan untuk ticker tersebut.")
else:
    # =============================
    # PREPROCESS DATA (seperti di kode Anda)
    # =============================
    df = data.copy()
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df[['Close']]
    df = df.reset_index()
    df.columns = ['Date', 'Close']
    df.index = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    feature_cols = ["Close"]
    target_col = "Close"
    window = 1

    data_features = df[feature_cols].values
    data_target = df[[target_col]].values

    values = df[['Close']].values
    n = len(values)
    n_train = int(n * 0.8)

    train_values = values[:n_train]
    test_values = values[n_train:]

    # Normalisasi Data
    scaler_X = MinMaxScaler().fit(data_features[:n_train])
    scaler_y = MinMaxScaler().fit(data_target[:n_train])

    Xs = scaler_X.transform(data_features)
    ys = scaler_y.transform(data_target)

    # Lagged Data set
    def make_sequences(X_scaled, y_scaled, window):
        X_seq, y_seq = [], []
        for i in range(window, len(X_scaled)):
            X_seq.append(X_scaled[i-window:i])
            y_seq.append(y_scaled[i])
        return np.array(X_seq), np.array(y_seq)

    X_seq_all, y_seq_all = make_sequences(Xs, ys, window=window)

    train_end_idx = n_train - window

    X_train = X_seq_all[:train_end_idx]
    y_train = y_seq_all[:train_end_idx]

    X_test = X_seq_all[train_end_idx:]
    y_test = y_seq_all[train_end_idx:]

    BASE_UNITS = 16
    BASE_DROPOUT = 0.5
    BASE_BATCH = 64
    BASE_EPOCHS = 100
    BASE_LR = 0.001

    def build_lstm_model(input_shape, units=16, dropout=0.01, lr=1e-3):
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(LSTM(units=units, input_shape=input_shape))
        if dropout > 0:
            model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    def smape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        num = np.abs(y_pred - y_true)
        den = (np.abs(y_true) + np.abs(y_pred)) / 2
        return np.mean(num / (den + 1e-8)) * 100

    def set_seed(seed_value):
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)

    # =============================
    # TRAIN MODELS (cached)
    # =============================
    @st.cache_resource
    def train_baseline():
        set_seed(42)
        model_base = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=BASE_UNITS,
            dropout=BASE_DROPOUT,
            lr=BASE_LR
        )

        history_base = model_base.fit(
            X_train, y_train,
            epochs=BASE_EPOCHS,
            batch_size=BASE_BATCH,
            validation_split=0.2,
            verbose=0
        )

        y_pred_scaled_base = model_base.predict(X_test, verbose=0)
        y_pred_base = scaler_y.inverse_transform(y_pred_scaled_base).flatten()
        y_true_base = scaler_y.inverse_transform(y_test).flatten()

        base_mape = mape(y_true_base, y_pred_base)
        base_smape = smape(y_true_base, y_pred_base)

        return model_base, history_base, base_mape, base_smape, y_pred_base, y_true_base

    @st.cache_resource
    def train_pso():

        PSO_N_PARTICLES = 10
        PSO_ITERS = 10
        PSO_OPTIONS = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}
        PSO_BOUNDS = ([16, 0.0001, 8, 0.1], [160, 0.001, 256, 1])
    
        val_frac_for_pso = 0.2
        n_tr_samples = X_train.shape[0]
        n_tr_val = int(n_tr_samples * (1 - val_frac_for_pso))
    
        X_tr_for_pso = X_train[:n_tr_val]
        y_tr_for_pso = y_train[:n_tr_val]
        X_val_for_pso = X_train[n_tr_val:]
        y_val_for_pso = y_train[n_tr_val:]
    
        def make_pso_obj(X_tr, y_tr, X_va, y_va, scaler_y):
            def obj_fn(particles):
                n_particles = particles.shape[0]
                costs = np.zeros(n_particles)
    
                for i, p in enumerate(particles):
                    units = int(np.round(p[0]))
                    lr = float(p[1])
                    batch = int(np.round(p[2]))
                    dropout = float(p[3])
    
                    try:
                        set_seed(42)
                        tf.keras.backend.clear_session()
    
                        model = build_lstm_model(
                            input_shape=(X_tr.shape[1], X_tr.shape[2]),
                            units=units,
                            dropout=dropout,
                            lr=lr
                        )
    
                        model.fit(X_tr, y_tr, epochs=10, batch_size=batch, verbose=0)
    
                        yv_pred = model.predict(X_va, verbose=0)
                        yv_pred_orig = scaler_y.inverse_transform(yv_pred).flatten()
                        yv_true_orig = scaler_y.inverse_transform(y_va).flatten()
    
                        costs[i] = mean_squared_error(yv_true_orig, yv_pred_orig)
    
                    except:
                        costs[i] = 1e12
    
                return costs
            return obj_fn
    
        pso_obj = make_pso_obj(X_tr_for_pso, y_tr_for_pso, X_val_for_pso, y_val_for_pso, scaler_y)
    
        optimizer = GlobalBestPSO(
            n_particles=PSO_N_PARTICLES,
            dimensions=4,
            options=PSO_OPTIONS,
            bounds=PSO_BOUNDS
        )
    
        # ===== INIT PBEST =====
        optimizer.swarm.pbest_cost = np.full(PSO_N_PARTICLES, np.inf)
        optimizer.swarm.pbest_pos = optimizer.swarm.position.copy()
        optimizer.swarm.best_cost = np.inf
        optimizer.swarm.best_pos = optimizer.swarm.position[0].copy()
    
        history_gbest_cost = []
        history_gbest_pos = []
    
        for it in range(PSO_ITERS):
    
            costs = pso_obj(optimizer.swarm.position)
    
            mask = costs < optimizer.swarm.pbest_cost
            optimizer.swarm.pbest_cost[mask] = costs[mask]
            optimizer.swarm.pbest_pos[mask] = optimizer.swarm.position[mask]
    
            best_idx = np.argmin(optimizer.swarm.pbest_cost)
            optimizer.swarm.best_cost = optimizer.swarm.pbest_cost[best_idx]
            optimizer.swarm.best_pos = optimizer.swarm.pbest_pos[best_idx]
    
            history_gbest_cost.append(float(optimizer.swarm.best_cost))
            history_gbest_pos.append(optimizer.swarm.best_pos.copy())
    
            r1 = np.random.rand(*optimizer.swarm.position.shape)
            r2 = np.random.rand(*optimizer.swarm.position.shape)
    
            optimizer.swarm.velocity = (
                PSO_OPTIONS['w'] * optimizer.swarm.velocity
                + PSO_OPTIONS['c1'] * r1 * (optimizer.swarm.pbest_pos - optimizer.swarm.position)
                + PSO_OPTIONS['c2'] * r2 * (optimizer.swarm.best_pos - optimizer.swarm.position)
            )
    
            optimizer.swarm.position += optimizer.swarm.velocity
    
            lb, ub = np.array(PSO_BOUNDS[0]), np.array(PSO_BOUNDS[1])
            optimizer.swarm.position = np.clip(optimizer.swarm.position, lb, ub)
    
        best_pos = history_gbest_pos[-1]
    
        best_units = int(np.round(best_pos[0]))
        best_lr = float(best_pos[1])
        best_batch = int(np.round(best_pos[2]))
        best_dropout = float(best_pos[3])
    
        set_seed(42)
    
        model_final = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=best_units,
            dropout=best_dropout,
            lr=best_lr
        )
    
        history_final = model_final.fit(
            X_train, y_train,
            epochs=100,
            batch_size=best_batch,
            validation_split=0.2,
            verbose=0
        )
    
        y_pred_scaled_final = model_final.predict(X_test, verbose=0)
        y_pred_final = scaler_y.inverse_transform(y_pred_scaled_final).flatten()
        y_true_final = scaler_y.inverse_transform(y_test).flatten()
    
        pso_mape = mape(y_true_final, y_pred_final)
        pso_smape = smape(y_true_final, y_pred_final)
    
        return model_final, history_final, pso_mape, pso_smape, y_pred_final, y_true_final, history_gbest_cost


    @st.cache_resource
    def train_ga():
        POP_SIZE = 10
        N_GENERATIONS = 10
        MUTATION_RATE = 0.1
        GA_LB = [16, 8, 0.1, 0.0001]
        GA_UB = [160, 256, 1, 0.001]

        def init_individual(lb, ub):
            return {
                'units': int(np.random.randint(lb[0], ub[0] + 1)),
                'batch_size': int(np.random.randint(lb[1], ub[1] + 1)),
                'dropout': float(np.random.uniform(lb[2], ub[2])),
                'lr': float(10 ** np.random.uniform(np.log10(lb[3]), np.log10(ub[3])))
            }

        def fitness_ga(indiv, X_tr, y_tr, X_val, y_val, scaler_y):
            try:
                units = int(np.round(indiv['units']))
                batch = int(np.round(indiv['batch_size']))
                dropout = float(indiv['dropout'])
                lr = float(indiv['lr'])
                epochs_fixed = 10
                set_seed(42)
                tf.keras.backend.clear_session()
                model = build_lstm_model(
                    input_shape=(X_tr.shape[1], X_tr.shape[2]),
                    units=units,
                    dropout=dropout,
                    lr=lr
                )
                model.fit(X_tr, y_tr, epochs=epochs_fixed, batch_size=batch, verbose=0)
                yv_pred = model.predict(X_val, verbose=0)
                yv_pred_orig = scaler_y.inverse_transform(yv_pred).flatten()
                yv_true_orig = scaler_y.inverse_transform(y_val).flatten()
                mse_val = mean_squared_error(yv_true_orig, yv_pred_orig)
                tf.keras.backend.clear_session()
                return mse_val
            except:
                tf.keras.backend.clear_session()
                return 1e12

        def mutate(indiv, lb, ub, rate):
            child = copy.deepcopy(indiv)
            if np.random.rand() < rate:
                child['units'] = int(np.random.randint(lb[0], ub[0] + 1))
            if np.random.rand() < rate:
                child['batch_size'] = int(np.random.randint(lb[1], ub[1] + 1))
            if np.random.rand() < rate:
                child['dropout'] = float(np.random.uniform(lb[2], ub[2]))
            if np.random.rand() < rate:
                child['lr'] = float(10 ** np.random.uniform(np.log10(lb[3]), np.log10(ub[3])))
            return child
        
        
        def crossover(p1, p2, alpha=0.25):
            """
            Extended Intermediate Crossover
            """
            child = {}
            for k in p1.keys():
                val1 = p1[k]
                val2 = p2[k]
        
                lower = min(val1, val2) - alpha * abs(val1 - val2)
                upper = max(val1, val2) + alpha * abs(val1 - val2)
        
                new_val = np.random.uniform(lower, upper)
        
                if k in ['units', 'batch_size']:
                    child[k] = int(np.round(new_val))
                else:
                    child[k] = float(new_val)
        
            return child
        

        val_frac_for_pso = 0.2
        n_tr_samples = X_train.shape[0]
        n_tr_val = int(n_tr_samples * (1 - val_frac_for_pso))
        X_tr_for_pso = X_train[:n_tr_val]
        y_tr_for_pso = y_train[:n_tr_val]
        X_val_for_pso = X_train[n_tr_val:]
        y_val_for_pso = y_train[n_tr_val:]

        population = [init_individual(GA_LB, GA_UB) for _ in range(POP_SIZE)]
        best_mse_ga = np.inf
        best_params_ga = None
        gbest_history_ga = []

        for gen in range(N_GENERATIONS):
            fitness_scores = [
                fitness_ga(ind, X_tr_for_pso, y_tr_for_pso, X_val_for_pso, y_val_for_pso, scaler_y)
                for ind in population
            ]
            order = np.argsort(fitness_scores)
            population = [population[i] for i in order]
            if fitness_scores[order[0]] < best_mse_ga:
                best_mse_ga = fitness_scores[order[0]]
                best_params_ga = population[0]
            gbest_history_ga.append(best_mse_ga)
            elites = population[:5]
            offspring = []
            while len(offspring) < POP_SIZE - len(elites):
                p1, p2 = np.random.choice(elites, 2, replace=False)
                child = crossover(p1, p2)
                child = mutate(child, GA_LB, GA_UB, MUTATION_RATE)
                offspring.append(child)
            population = elites + offspring

        best_units_ga = int(np.round(best_params_ga['units']))
        best_lr_ga = float(best_params_ga['lr'])
        best_batch_ga = int(np.round(best_params_ga['batch_size']))
        best_dropout_ga = float(best_params_ga['dropout'])

        set_seed(42)
        final_model_ga = build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            units=best_units_ga,
            dropout=best_dropout_ga,
            lr=best_lr_ga
        )

        history_ga = final_model_ga.fit(
            X_train, y_train,
            epochs=100,
            batch_size=best_batch_ga,
            validation_split=0.2,
            verbose=0
        )

        y_pred_scaled_ga = final_model_ga.predict(X_test, verbose=0)
        y_pred_ga = scaler_y.inverse_transform(y_pred_scaled_ga).flatten()
        y_true_ga = scaler_y.inverse_transform(y_test).flatten()

        ga_mape = mape(y_true_ga, y_pred_ga)
        ga_smape = smape(y_true_ga, y_pred_ga)

        return final_model_ga, history_ga, ga_mape, ga_smape, y_pred_ga, y_true_ga, gbest_history_ga

   
    # =========================================================
    # SESSION STATE (agar tidak retrain saat pindah tab)
    # =========================================================
    if "trained" not in st.session_state:
        st.session_state.trained = False
    
    # =========================================================
    # BUTTON TRAIN MODEL
    # =========================================================
    st.sidebar.markdown("### Training Model")

    if st.sidebar.button("Run Training Model"):
    
        progress_bar = st.progress(0)
        status_text = st.empty()
    
        with st.spinner("Training models..."):
    
            # ================= BASELINE =================
            status_text.write("Training Baseline LSTM...")
            st.session_state.model_base, \
            st.session_state.history_base, \
            st.session_state.base_mape, \
            st.session_state.base_smape, \
            st.session_state.y_pred_base, \
            st.session_state.y_true_base = train_baseline()
    
            progress_bar.progress(33)
    
            # ================= PSO =================
            status_text.write("Training PSO-Optimized LSTM...")
            st.session_state.model_pso, \
            st.session_state.history_pso, \
            st.session_state.pso_mape, \
            st.session_state.pso_smape, \
            st.session_state.y_pred_pso, \
            st.session_state.y_true_pso, \
            st.session_state.pso_gbest = train_pso()
    
            progress_bar.progress(66)
    
            # ================= GA =================
            status_text.write("Training GA-Optimized LSTM...")
            st.session_state.model_ga, \
            st.session_state.history_ga, \
            st.session_state.ga_mape, \
            st.session_state.ga_smape, \
            st.session_state.y_pred_ga, \
            st.session_state.y_true_ga, \
            st.session_state.ga_gbest = train_ga()
    
            progress_bar.progress(100)
    
            status_text.write("Training selesai.")
            st.session_state.trained = True
    
        st.success("Semua model berhasil dilatih.")
                
    # =============================
    # SECTION 1 : INFORMASI DATA
    # =============================
    if section == "Informasi Data":
        st.subheader(f"Pergerakan Harga Saham {ticker}")
        st.line_chart(data['Close'])

        st.subheader("Statistik Deskriptif (Close)")
        st.write(data['Close'].describe())

    # =============================
    # SECTION 2 : IN DEPTH ANALYSIS
    # =============================
    elif section == "In-Depth Analysis":

        if not st.session_state.trained:
            st.warning("Klik 'Run Training Model' terlebih dahulu.")
        else:
            history_base = st.session_state.history_base
            history_pso = st.session_state.history_pso
            history_ga = st.session_state.history_ga
    
            # =====================================================
            # VALIDATION LOSS (3 garis dalam 1 grafik)
            # =====================================================
            col1, col2, col3 = st.columns(3)

            with col1:
                fig1, ax1 = plt.subplots()
                ax1.plot(history_base.history['loss'])
                ax1.plot(history_base.history['val_loss'])
                ax1.set_title('Baseline LSTM')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend(['Training Loss','Validation Loss'])
                st.pyplot(fig1, use_container_width=True)
            
            with col2:
                fig2, ax2 = plt.subplots()
                ax2.plot(history_ga.history['loss'])
                ax2.plot(history_ga.history['val_loss'])
                ax2.set_title('GA-LSTM')
                ax2.set_xlabel('Epoch')
                ax2.legend(['Training Loss','Validation Loss'])
                st.pyplot(fig2, use_container_width=True)
            
            with col3:
                fig3, ax3 = plt.subplots()
                ax3.plot(history_pso.history['loss'])
                ax3.plot(history_pso.history['val_loss'])
                ax3.set_title('PSO-LSTM')
                ax3.set_xlabel('Epoch')
                ax3.legend(['Training Loss','Validation Loss'])
                st.pyplot(fig3, use_container_width=True)
    
            # =====================================================
            # ACTUAL VS PREDICTED (3 MODEL)
            # =====================================================
            st.subheader("Actual vs Predicted Comparison")
    
            fig4, ax4 = plt.subplots()
            ax4.plot(st.session_state.y_true_base, label="Actual", linewidth=2)
            ax4.plot(st.session_state.y_pred_base, label="Baseline")
            ax4.plot(st.session_state.y_pred_pso, label="PSO")
            ax4.plot(st.session_state.y_pred_ga, label="GA")
            ax4.legend()
            
            st.pyplot(fig4, use_container_width=True)

            # =====================================================
            # MAPE TABLE
            # =====================================================
            st.subheader("MAPE Comparison")
    
            results = pd.DataFrame({
                "Model": ["Baseline", "PSO", "GA"],
                "MAPE": [
                    st.session_state.base_mape,
                    st.session_state.pso_mape,
                    st.session_state.ga_mape
                ]
            })
    
            st.dataframe(results)
    
        

    # =========================================================
    # SECTION 3 : HASIL FORECAST
    # =========================================================
    elif section == "Hasil Forecast":

        if not st.session_state.trained:
            st.warning("Klik 'Run Training Model' terlebih dahulu.")
        else:
    
            st.subheader("Forecast Future")
    
            future_days = st.slider("Forecast horizon (hari)", 5, 30, 7)
    
            last_window = X_test[-1].copy()
            future_preds = []
    
            model = st.session_state.model_base
    
            for _ in range(future_days):
                pred = model.predict(last_window.reshape(1, last_window.shape[0], last_window.shape[1]), verbose=0)
                future_preds.append(pred[0,0])
    
                last_window = np.roll(last_window, -1)
                last_window[-1] = pred
    
            future_preds = scaler_y.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
    
            # ===============================
            # Grafik forecast
            # ===============================
            fig, ax = plt.subplots()
            ax.plot(future_preds, label="Forecast")
            ax.set_title("Future Forecast")
            ax.legend()
            st.pyplot(fig, use_container_width=True)

    
            # ===============================
            # tabel forecast
            # ===============================
            future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=future_days)
    
            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Forecast": future_preds
            })
    
            st.dataframe(forecast_df)
