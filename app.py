import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from datetime import timedelta
import copy
from pyswarms.single.global_best import GlobalBestPSO
import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"

# Set Seed
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

st.set_page_config(layout="wide")

st.markdown("""
<style>

/* Padding halaman */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 1rem;
    max-width: 1000px;   /* batasi lebar konten */
}

/* kecilkan font global */
html {
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =============================
# FUNGSI PLOT KECIL CENTER
# =============================
def show_plot(fig, ratio=[1,5,1]):
    left, center, right = st.columns(ratio)
    with center:
        st.pyplot(fig)

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.title("Input Data Saham (Excel)")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel (Kolom: Date & Close)",
    type=["xlsx"]
)

section = st.sidebar.radio(
    "Menu",
    ["Informasi Data", "Training & Evaluasi", "Forecast"]
)

# =============================
# LOAD DATA
# =============================
def load_excel(file):
    df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]

    if "Date" not in df.columns or "Close" not in df.columns:
        st.error("File harus memiliki kolom: Date dan Close")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df = df[["Date", "Close"]].dropna().reset_index(drop=True)
    return df

if uploaded_file is None:
    st.warning("Silakan upload file Excel terlebih dahulu.")
    st.stop()

# Load data
df = load_excel(uploaded_file)

# Preprocessing
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

scaler_X = MinMaxScaler().fit(data_features[:n_train])
scaler_y = MinMaxScaler().fit(data_target[:n_train])

Xs = scaler_X.transform(data_features)
ys = scaler_y.transform(data_target)

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

# Pemodelan
def build_lstm_model(input_shape, units, dropout, lr):
    K.clear_session()
    model = Sequential()
    model.add(LSTM(units=units, input_shape=input_shape))
    if dropout is not None and dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="mse")

    return model  
        
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def set_seed(seed=42):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed))
    
# =============================
# BASELINE TRAIN
# =============================
def train_baseline():
    set_seed(42)
    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        units=16,
        dropout=0.5,
        lr=0.001
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        verbose=0
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()

    return model, history, mape(y_true, y_pred), y_pred, y_true
    
def train_ga():
    set_seed(42)
        
    POP_SIZE = 10
    N_GENERATIONS = 10
    MUTATION_RATE = 0.3
    GA_LB = [16, 8, 0.1, 0.0001]
    GA_UB = [160, 256, 0.8, 0.001]

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
            
            set_seed(42)
            K.clear_session()
            
            model = build_lstm_model_ga(
                units=units,
                dropout=dropout,
                lr=lr,
                input_shape=(X_tr.shape[1], X_tr.shape[2])
            )
            model.fit(
                X_tr, y_tr,
                epochs=20,
                batch_size=batch,
                verbose=0
            )
            yv_pred = model.predict(X_val, verbose=0)
            yv_pred_orig = scaler_y.inverse_transform(yv_pred).flatten()
            yv_true_orig = scaler_y.inverse_transform(y_val).flatten()
        
            # Optimasi berdasarkan MSE
            mse_val = mean_squared_error(yv_true_orig, yv_pred_orig)
            tf.keras.backend.clear_session()
            return mse_val
            
        except Exception as e:
            print(f"GA Eval Error: {e}")
            tf.keras.backend.clear_session()
            return 1e12

    def crossover(p1, p2, alpha=0.25):
        child = {}
        for k in p1.keys():
            val1 = p1[k]
            val2 = p2[k]
            lower = min(val1, val2) - alpha * abs(val1 - val2)
            upper = max(val1, val2) + alpha * abs(val1 - val2)
            new_val = np.random.uniform(lower, upper)
                      
            if k == 'units':
                child[k] = int(np.clip(np.round(new_val), GA_LB[0], GA_UB[0]))
            elif k == 'batch_size':
                child[k] = int(np.clip(np.round(new_val), GA_LB[1], GA_UB[1]))
            elif k == 'dropout':
                child[k] = float(np.clip(new_val, GA_LB[2], GA_UB[2]))
            elif k == 'lr':
                child[k] = float(np.clip(new_val, GA_LB[3], GA_UB[3]))
        return child
        
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
        
    val_frac_for_ga = 0.2
    n_tr_samples_ga = X_train.shape[0]
    n_tr_val_ga = int(n_tr_samples_ga * (1 - val_frac_for_ga))
    
    X_tr_for_ga = X_train[:n_tr_val_ga]
    y_tr_for_ga = y_train[:n_tr_val_ga]
    X_val_for_ga = X_train[n_tr_val_ga:]
    y_val_for_ga = y_train[n_tr_val_ga:]

    # Membangun Model LSTM-GA
    def build_lstm_model_ga(units, dropout, lr, input_shape):
    
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(LSTM(units=units, input_shape=input_shape))
        if dropout > 0:
            model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model
        
    population = [init_individual(GA_LB, GA_UB) for _ in range(POP_SIZE)]
    best_mse_ga = np.inf
    best_params_ga = None
    gbest_history_ga = []

    for gen in range(N_GENERATIONS):
        fitness_scores = [
            fitness_ga(ind, X_tr_for_ga, y_tr_for_ga, X_val_for_ga, y_val_for_ga, scaler_y)
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
    final_model_ga = build_lstm_model_ga(
        units=best_units_ga,
        dropout=best_dropout_ga,
        lr=best_lr_ga,
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
    history_ga = final_model_ga.fit(
        X_train, y_train,
        epochs=100,
        batch_size=best_batch_ga,
        validation_split=0.2,
        verbose=0
    )

    y_pred_scaled_ga = final_model_ga.predict(X_test)
    y_pred_ga = scaler_y.inverse_transform(y_pred_scaled_ga).flatten()
    y_true_ga = scaler_y.inverse_transform(y_test).flatten()

    ga_mape = mape(y_true_ga, y_pred_ga)
    return final_model_ga, history_ga, ga_mape, y_pred_ga, y_true_ga, gbest_history_ga
    

def train_pso():

    set_seed(42)
    K.clear_session()

    # =========================
    # KONFIGURASI PSO
    # =========================
    PSO_N_PARTICLES = 10
    PSO_ITERS = 10
    PSO_OPTIONS = {'c1': 1.5, 'c2': 1.5, 'w': 0.5}

    PSO_BOUNDS = (
        np.array([16, 0.0001, 8, 0.1]),
        np.array([160, 0.001, 256, 0.8])
    )

    # =========================
    # SPLIT DATA UNTUK PSO
    # =========================
    val_frac_for_pso = 0.2
    n_tr_samples = X_train.shape[0]
    n_tr_val = int(n_tr_samples * (1 - val_frac_for_pso))

    X_tr_for_pso = X_train[:n_tr_val]
    y_tr_for_pso = y_train[:n_tr_val]
    X_val_for_pso = X_train[n_tr_val:]
    y_val_for_pso = y_train[n_tr_val:]

    # =========================
    # OBJECTIVE FUNCTION (MSE)
    # =========================
    def make_pso_obj(X_tr, y_tr, X_va, y_va, scaler_y):
        def obj_fn(particles):
            n_particles = particles.shape[0]
            costs = np.zeros(n_particles)

            for i, p in enumerate(particles):

                units = int(np.round(p[0]))
                lr = float(p[1])
                batch = int(np.round(p[2]))
                dropout = float(p[3])
                epochs_fixed = 20

                try:
                    set_seed(42)
                    K.clear_session()

                    model = build_lstm_model(
                        input_shape=(X_tr.shape[1], X_tr.shape[2]),
                        units=units,
                        dropout=dropout,
                        lr=lr
                    )

                    model.fit(
                        X_tr, y_tr,
                        epochs=epochs_fixed,
                        batch_size=batch,
                        verbose=0
                    )

                    yv_pred = model.predict(X_va, verbose=0)
                    yv_pred_orig = scaler_y.inverse_transform(yv_pred).flatten()
                    yv_true_orig = scaler_y.inverse_transform(y_va).flatten()

                    costs[i] = mean_squared_error(yv_true_orig, yv_pred_orig)

                except Exception as e:
                    print("PSO eval error:", e)
                    costs[i] = 1e12

                K.clear_session()

            return costs
        return obj_fn


    pso_obj = make_pso_obj(
        X_tr_for_pso, y_tr_for_pso,
        X_val_for_pso, y_val_for_pso,
        scaler_y
    )

    # =========================
    # INIT OPTIMIZER
    # =========================
    optimizer = GlobalBestPSO(
        n_particles=PSO_N_PARTICLES,
        dimensions=4,
        options=PSO_OPTIONS,
        bounds=PSO_BOUNDS
    )

    n_particles, dims = optimizer.swarm.position.shape
    optimizer.swarm.pbest_pos = optimizer.swarm.position.copy()
    optimizer.swarm.pbest_cost = np.full(n_particles, np.inf)

    history_gbest_cost = []
    history_gbest_pos = []

    # =========================
    # LOOP PSO MANUAL
    # =========================
    for it in range(PSO_ITERS):

        costs = pso_obj(optimizer.swarm.position)

        mask = costs < optimizer.swarm.pbest_cost
        optimizer.swarm.pbest_cost[mask] = costs[mask]
        optimizer.swarm.pbest_pos[mask] = optimizer.swarm.position[mask].copy()

        best_idx = np.argmin(optimizer.swarm.pbest_cost)
        optimizer.swarm.best_cost = optimizer.swarm.pbest_cost[best_idx]
        optimizer.swarm.best_pos = optimizer.swarm.pbest_pos[best_idx].copy()

        history_gbest_cost.append(float(optimizer.swarm.best_cost))
        history_gbest_pos.append(optimizer.swarm.best_pos.copy())

        # update velocity
        r1 = np.random.rand(*optimizer.swarm.position.shape)
        r2 = np.random.rand(*optimizer.swarm.position.shape)

        optimizer.swarm.velocity = (
            PSO_OPTIONS['w'] * optimizer.swarm.velocity
            + PSO_OPTIONS['c1'] * r1 * (optimizer.swarm.pbest_pos - optimizer.swarm.position)
            + PSO_OPTIONS['c2'] * r2 * (optimizer.swarm.best_pos - optimizer.swarm.position)
        )

        # update posisi
        optimizer.swarm.position += optimizer.swarm.velocity
        lb, ub = PSO_BOUNDS
        optimizer.swarm.position = np.clip(optimizer.swarm.position, lb, ub)

    # =========================
    # PARAMETER TERBAIK
    # =========================
    best_pos = history_gbest_pos[-1]
    best_units = int(np.round(best_pos[0]))
    best_lr = float(best_pos[1])
    best_batch = int(np.round(best_pos[2]))
    best_dropout = float(best_pos[3])

    # =========================
    # TRAIN FINAL MODEL
    # =========================
    set_seed(42)
    K.clear_session()

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

    # =========================
    # EVALUASI TEST
    # =========================
    y_pred_scaled = model_final.predict(X_test, verbose=0)

    y_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()

    pso_mape = mape(y_true, y_pred)

    return (
        model_final,
        history_final,
        pso_mape,
        y_pred,
        y_true,
        np.array(history_gbest_cost)
    )
    
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
    with st.spinner("Training Baseline, GA, PSO"):
        (st.session_state.model_base,
         st.session_state.history_base,
         st.session_state.base_mape,
         st.session_state.y_pred_base,
         st.session_state.y_true_base) = train_baseline()

        (st.session_state.model_ga,
         st.session_state.history_ga,
         st.session_state.ga_mape,
         st.session_state.y_pred_ga,
         st.session_state.y_true_ga,
         st.session_state.gbest_ga) = train_ga()
                
        (st.session_state.model_pso,
         st.session_state.history_pso,
         st.session_state.pso_mape,
         st.session_state.y_pred_pso,
         st.session_state.y_true_pso,
         st.session_state.gbest_pso) = train_pso()

        st.session_state.trained = True
        st.success("Training selesai!")
                
# =============================
# SECTION 1 : INFORMASI DATA
# =============================
if section == "Informasi Data":
    st.subheader("Grafik Harga Saham")
    fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df["Date"], df["Close"])
    ax.set_title("Pergerakan Harga Saham")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    show_plot(fig)
    
    st.subheader("Statistik Deskriptif")

    left, center, right = st.columns([1,5,1])
    with center:
        st.dataframe(df["Close"].describe().to_frame())
        
# =============================
# SECTION 2 : TRAINING & EVALUASI
# =============================
elif section == "Training & Evaluasi":
    if not st.session_state.trained:
        st.warning("Klik 'Run Training Model' terlebih dahulu.")
    else:
        history_base = st.session_state.history_base
        history_pso = st.session_state.history_pso
        history_ga = st.session_state.history_ga

        st.subheader("Training vs Validation Loss")
        
        # =====================================================
        # VALIDATION LOSS (3 garis dalam 1 grafik)
        # =====================================================
        col1, col2, col3 = st.columns(3)

        # BASELINE
        with col1:
            fig1, ax1 = plt.subplots(figsize=(3,2))
            ax1.plot(history_base.history['loss'], label='Train')
            ax1.plot(history_base.history['val_loss'], label='Val')
            ax1.set_title('Baseline LSTM')
            ax1.legend(fontsize=8)
            st.pyplot(fig1)
        
        # GA
        with col2:
            fig2, ax2 = plt.subplots(figsize=(3,2))
            ax2.plot(history_ga.history['loss'], label='Train')
            ax2.plot(history_ga.history['val_loss'], label='Val')
            ax2.set_title('GA-LSTM')
            ax2.legend(fontsize=8)
            st.pyplot(fig2)
    
        # PSO
        with col3:
            fig3, ax3 = plt.subplots(figsize=(3,2))
            ax3.plot(history_pso.history['loss'], label='Train')
            ax3.plot(history_pso.history['val_loss'], label='Val')
            ax3.set_title('PSO-LSTM')
            ax3.legend(fontsize=8)
            st.pyplot(fig3)
        
        # =====================================================
        # ACTUAL VS PREDICTED (3 MODEL)
        # =====================================================
        st.subheader("Actual vs Predicted Comparison")

        fig4, ax4 = plt.subplots(figsize=(6,3))
        ax4.plot(st.session_state.y_true_base, label="Actual", linewidth=2)
        ax4.plot(st.session_state.y_pred_base, label="Baseline")
        ax4.plot(st.session_state.y_pred_pso, label="PSO")
        ax4.plot(st.session_state.y_pred_ga, label="GA")
        ax4.legend(fontsize=8)
        ax4.set_title("Actual vs Predicted", fontsize=10)
        
        show_plot(fig4)

        
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
elif section == "Forecast":
    if not st.session_state.trained:
        st.warning("Klik Run Training Model terlebih dahulu.")
    else:
        future_days = st.slider("Forecast (hari)", 5, 30, 7)

        last_window = X_test[-1].copy()
        future_preds = []

        model = st.session_state.model_pso  
        
        for _ in range(future_days):
            pred = model.predict(last_window.reshape(1,1,1), verbose=0)
            future_preds.append(pred[0,0])
            last_window = pred.reshape(1,1,1)

        future_preds = scaler_y.inverse_transform(np.array(future_preds).reshape(-1,1)).flatten()
        # ===============================
        # BUAT TANGGAL MASA DEPAN
        # ===============================
        future_dates = pd.bdate_range(
            start=df["Date"].iloc[-1],
            periods=future_days + 1
        )[1:]

        # ===============================
        # Grafik forecast
        # ===============================
        st.subheader("Forecast Harga Saham")

        fig, ax = plt.subplots(figsize=(9,3))

        # data historis
        ax.plot(df["Date"], df["Close"], label="Data Historis", linewidth=2)
        
        # sambungan garis terakhir (optional)
        ax.plot(
            [df["Date"].iloc[-1], future_dates[0]],
            [df["Close"].iloc[-1], future_preds[0]],
            linestyle="--"
        )
        
        # forecast
        ax.plot(
            future_dates,
            future_preds,
            label="Forecast",
            linestyle="--",
            marker="o"
        )
        
        ax.legend()
        ax.set_title("Pergerakan Harga Saham + Forecast")
        
        st.pyplot(fig)

        # ===============================
        # tabel forecast
        # ===============================
        future_dates = pd.date_range(
            start=df["Date"].iloc[-1] + timedelta(days=1),
            periods=future_days
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": future_preds
        })

        st.dataframe(forecast_df)

